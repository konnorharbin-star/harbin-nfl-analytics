"""Immutable prospective recommendation receipts, separate from research signals.

A local receipt is pending publication. Communicate BET only after verifying that
its exact bytes exist in a successful GitHub commit before kickoff. Never backfill.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


def timestamp(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return dt.astimezone(UTC)


def publication_is_pregame(captured_at, committed_at, kickoff):
    """Git commit time has one-second precision; capture retains full precision."""
    capture, commit, start = map(timestamp, (captured_at, committed_at, kickoff))
    return capture < start and capture.replace(microsecond=0) <= commit < start


def finite(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("Nonfinite number")
    return n


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def archive_decision(row, *, gate, root, observed_at=None):
    """Persist exact board recommendation, failing closed on incomplete BET evidence."""
    now = observed_at or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("Observation requires timezone")
    if not row.get("game_id") or row.get("sport") not in {"nfl", "cfb"}:
        raise ValueError("Missing sport/game identity")
    approved = row.get("bet_approved") is True and row.get("decision") == "BET_READY"
    blockers = [x for x in str(row.get("reasons") or "").split(";") if x]
    if approved:
        required = ("model_version", "source_url", "quote_observed_at", "confidence", "risks")
        if any(not row.get(k) for k in required):
            blockers.append("RECOMMENDATION_PROVENANCE_INCOMPLETE")
        if gate.get("production_eligible") is not True:
            blockers.append("GLOBAL_RELEASE_BLOCKED")
        try:
            kickoff = timestamp(row["kickoff"])
            quote = timestamp(row["quoted_at"])
            observed = timestamp(row["quote_observed_at"])
            if not quote <= observed <= now < kickoff or (now - quote).total_seconds() > 7200:
                blockers.append("INVALID_RECOMMENDATION_TIMING")
        except (ValueError, KeyError, TypeError):
            blockers.append("INVALID_RECOMMENDATION_TIMING")
        try:
            probability = finite(row["model_probability"])
            conservative = finite(row["conservative_probability"])
            odds = finite(row["american_odds"])
            push = finite(row.get("push_probability", 0))
            if not 0 < conservative <= probability < 1 or not 0 <= push < 1 - probability:
                raise ValueError("Invalid probability distribution")
            if abs(odds) < 100 or odds != int(odds):
                raise ValueError("Invalid odds")
            payout = odds / 100 if odds > 0 else 100 / -odds
            ev = probability * payout - (1 - probability - push)
            conservative_ev = conservative * payout - (1 - conservative - push)
            if conservative_ev <= 0:
                blockers.append("NO_CONSERVATIVE_EV")
        except (KeyError, ValueError, TypeError):
            blockers.append("INCOMPLETE_PRICE_DISTRIBUTION")
    decision = "BET" if approved and not blockers else "NO_BET"
    # Same quote/model/selection is one recommendation across repeated invocations.
    identity = {
        k: row.get(k)
        for k in (
            "sport",
            "game_id",
            "market",
            "side",
            "line",
            "american_odds",
            "book",
            "quoted_at",
            "model_version",
        )
    }
    identity["decision"] = decision
    if decision == "NO_BET":
        identity["decision_date"] = now.date().isoformat()
        identity["blockers"] = sorted(set(blockers))
    bet_id = hashlib.sha256(canonical(identity).encode()).hexdigest()
    directory = Path(root) / ("bets" if decision == "BET" else "no_bets")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{bet_id}.json"
    if target.exists():
        existing = json.loads(target.read_text())
        if existing["identity"] != identity:
            raise ValueError("Receipt identity collision")
        return existing
    record = {
        "schema_version": 1,
        "bet_id": bet_id,
        "identity": identity,
        "decision": decision,
        "recommended_at": now.astimezone(UTC).isoformat(),
        "original_board_row": row,
        "release_gate": gate,
        "blockers": sorted(set(blockers)),
        "stake_units": 1 if decision == "BET" else 0,
        "fill_status": "HYPOTHETICAL_QUOTE_NOT_ACTUAL_EXECUTION",
        "publication_status": "PENDING_GITHUB_COMMIT",
    }
    if decision == "BET":
        conditional = probability / (1 - push)
        record.update(
            {
                "calculated_ev": ev,
                "conservative_ev": conservative_ev,
                "fair_american_odds": -100 * conditional / (1 - conditional)
                if conditional >= 0.5
                else 100 * (1 - conditional) / conditional,
            }
        )
    # Publish fully flushed bytes atomically; readers never see partial JSON.
    encoded = canonical(record)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            return json.loads(target.read_text())
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record
