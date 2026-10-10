"""Audit prospective sportsbook quote evidence and frozen one-unit paper entries.

RESEARCH ONLY. Collector timestamps do not certify sportsbook prices. A source
verification flag is a human attestation, not cryptographic proof of a market.
No automatic bets, production recommendations or model updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

MAX_ORIGIN_LAG = timedelta(minutes=15)
LAST_QUOTE_WINDOW = timedelta(minutes=30)
HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def utc(value):
    if not isinstance(value, str):
        raise ValueError("An explicit aware ISO-8601 timestamp is required")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Naive timestamps cannot prove pregame timing")
    return dt.astimezone(UTC)


def decimal(american):
    if isinstance(american, bool):
        raise ValueError("Invalid American price")
    try:
        n = float(american)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid American price") from exc
    if not math.isfinite(n) or abs(n) < 100 or n != int(n):
        raise ValueError("Invalid American price")
    return 1 + (n / 100 if n > 0 else 100 / -n)


def pair_probability(side_odds, other_odds):
    a, b = 1 / decimal(side_odds), 1 / decimal(other_odds)
    return a / (a + b)


def source_reasons(row, *, observed_key="observed_at", source_key="reported_source_time"):
    """Machine-check consistency, but do not accept source flags as verification."""
    reasons = []
    for field in ("book_identity_verified", "source_quote_time_verified",
                  "executable_price_verified"):
        if row.get(field) is not True:
            reasons.append("UNVERIFIED_" + field.upper())
    if not isinstance(row.get("source_url"), str) or not row["source_url"].startswith(
        "https://"
    ):
        reasons.append("NO_DIRECT_SOURCE_URL")
    try:
        observed = utc(row[observed_key])
        origin = utc(row[source_key])
        kickoff = utc(row["kickoff"])
        if not origin <= observed < kickoff:
            reasons.append("INVALID_PREGAME_QUOTE_CHRONOLOGY")
        if observed - origin > MAX_ORIGIN_LAG:
            reasons.append("STALE_SOURCE_UPDATE")
    except (KeyError, ValueError, TypeError):
        reasons.append("UNKNOWN_OR_INVALID_ORIGIN_TIME")
    try:
        if row.get("market") not in {"moneyline", "spread", "total"}:
            raise ValueError("Invalid market")
        sides = row["sides"]
        if sides not in (["home", "away"], ["over", "under"]):
            raise ValueError("Invalid side order")
        if (row["market"] == "total") != (sides == ["over", "under"]):
            raise ValueError("Market/side mismatch")
        if row["market"] == "moneyline" and row.get("line") is not None:
            raise ValueError("Moneyline has no handicap")
        if row["market"] != "moneyline" and (
            row.get("line") is None or not math.isfinite(float(row["line"]))
        ):
            raise ValueError("Missing non-moneyline handicap")
        if len(row["odds"]) != 2:
            raise ValueError("Missing two-sided price")
        pair_probability(*row["odds"])
    except (KeyError, TypeError, ValueError, OverflowError):
        reasons.append("INVALID_PAIRED_MARKET")
    return sorted(set(reasons))


def capture_audit(capture_root):
    paths = sorted(Path(capture_root).glob("*.json"))
    seen = set()
    reasons = Counter()
    source_complete = 0
    valid_pairs = 0
    raw = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("source_records", []):
            raw += 1
            # Deduplicate identical captured observations repeated by later audits.
            key = hashlib.sha256(
                json.dumps(row, sort_keys=True, default=str).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            why = source_reasons(row)
            reasons.update(why)
            if "INVALID_PAIRED_MARKET" not in why:
                valid_pairs += 1
            if not why:
                source_complete += 1
    return {
        "captures": len(paths),
        "raw_source_records": raw,
        "distinct_source_records": len(seen),
        "valid_structural_pairs": valid_pairs,
        "source_evidence_complete": source_complete,
        "source_gate_rejections": dict(sorted(reasons.items())),
        "source_gate_status": (
            "NO_VERIFIED_PRICE_ORIGIN" if not source_complete
            else "ATTESTED_SOURCE_METADATA_PRESENT_MANUAL_REVIEW_REQUIRED"
        ),
        "source_evidence_is_independent_book_verification": False,
        "betting_authorized": False,
    }


def first_commit_time(path):
    """Require original tracked file bytes in a pregame commit, not a local edit.

    Git commit metadata is an imperfect timestamp. Independent provider/source
    provenance must still be checked and no betting authorization follows.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(path)],
            check=True, capture_output=True, text=True,
        )
        history = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H %cI", "--", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        if len(history) != 1:
            return None
        sha, authored = history[0].split(" ", 1)
        frozen = subprocess.run(
            ["git", "show", f"{sha}:{path.as_posix()}"],
            check=True, capture_output=True,
        ).stdout
        if frozen != path.read_bytes():
            return None
        return utc(authored)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def entry_reasons(row, published_at):
    reasons = []
    if row.get("game_id") in (None, "") or row.get("sport") not in {"nfl", "cfb"}:
        reasons.append("INVALID_GAME")
    if row.get("market") not in {"moneyline", "spread", "total"}:
        reasons.append("INVALID_MARKET")
    if row.get("side") not in {"home", "away", "over", "under"} or (
        row.get("market") == "total"
    ) != (row.get("side") in {"over", "under"}):
        reasons.append("INVALID_SIDE")
    try:
        if row.get("market") == "moneyline":
            if row.get("line") is not None:
                raise ValueError("Moneyline with handicap")
        elif row.get("line") is None or not math.isfinite(float(row["line"])):
            raise ValueError("Missing or invalid handicap")
        pair_probability(row["american_odds"], row["opposite_american_odds"])
    except (KeyError, ValueError, TypeError, OverflowError):
        reasons.append("INVALID_EXACT_PAIRED_PRICE")
    if not isinstance(row.get("book"), str) or not row["book"].strip():
        reasons.append("UNIDENTIFIED_BOOK")
    if not isinstance(row.get("source_url"), str) or not row["source_url"].startswith(
        "https://"
    ):
        reasons.append("MISSING_DIRECT_BOOK_URL")
    for field in (
        "book_identity_verified", "source_quote_time_verified",
        "executable_price_verified", "book_access_verified",
        "settlement_rules_verified",
    ):
        if row.get(field) is not True:
            reasons.append("UNVERIFIED_" + field.upper())
    try:
        origin, observed, frozen, kickoff = map(
            utc,
            (row["source_quote_at"], row["observed_at"], row["frozen_at"], row["kickoff"]),
        )
        if not origin <= observed <= frozen < kickoff:
            reasons.append("INVALID_CAPTURE_SEQUENCE")
        if observed - origin > MAX_ORIGIN_LAG:
            reasons.append("STALE_ORIGIN_QUOTE")
        if published_at is None or not observed <= published_at < kickoff:
            reasons.append("MISSING_VERIFIABLE_PREGAME_GIT_COMMIT")
    except (KeyError, ValueError, TypeError):
        reasons.append("UNKNOWN_CAPTURE_TIMESTAMPS")
    # A research model signal must be frozen independently of sportsbook prices.
    if not isinstance(row.get("frozen_model_sha256"), str) or not HEX_SHA256.fullmatch(
        row["frozen_model_sha256"]
    ):
        reasons.append("MISSING_FROZEN_MODEL_HASH")
    try:
        if not 0 < float(row["model_side_probability"]) < 1:
            raise ValueError("Invalid model probability")
    except (ValueError, KeyError, TypeError):
        reasons.append("INVALID_FROZEN_MODEL_PROBABILITY")
    return sorted(set(reasons))


def _read_by_id(directory):
    rows = {}
    for path in sorted(Path(directory).glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(row, dict) or str(row.get("receipt_id")) != path.stem:
            raise ValueError(f"Receipt ID must match immutable filename: {path}")
        if path.stem in rows:
            raise ValueError(f"Duplicate paper receipt ID: {path.stem}")
        rows[path.stem] = (row, path)
    return rows


def _result(row, final):
    if final.get("status") != "FINAL" or final.get("verified_result_source") is not True:
        return None
    if (row["sport"], row["game_id"]) != (final.get("sport"), final.get("game_id")):
        return None
    try:
        kickoff = utc(row["kickoff"])
        if utc(final["observed_at"]) <= kickoff:
            return None
        if not final["source_url"].startswith("https://"):
            return None
        home = int(final["home_score"])
        away = int(final["away_score"])
        if not 0 <= home <= 120 or not 0 <= away <= 120:
            return None
        if home != float(final["home_score"]) or away != float(final["away_score"]):
            return None
        market, side = row["market"], row["side"]
        if market == "moneyline":
            if home == away:
                return None
            margin = home - away if side == "home" else away - home
        elif market == "spread":
            margin = (home - away if side == "home" else away - home) + float(row["line"])
        else:
            margin = (home + away - float(row["line"])) * (1 if side == "over" else -1)
        outcome = "WIN" if margin > 0 else "LOSS" if margin < 0 else "PUSH"
        return outcome, (decimal(row["american_odds"]) - 1
                         if outcome == "WIN" else -1.0 if outcome == "LOSS" else 0.0)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _late_quote(row, close, pub):
    """Same book, market, selection and handicap only; never mix spread numbers."""
    if close is None:
        return None, "MISSING_LATE_QUOTE"
    fields = ("sport", "game_id", "market", "side", "book", "line", "kickoff")
    if any(row.get(k) != close.get(k) for k in fields):
        return None, "CHANGED_MARKET_HANDICAP_OR_BOOK"
    for field in (
        "book_identity_verified", "source_quote_time_verified",
        "executable_price_verified", "book_access_verified",
        "settlement_rules_verified",
    ):
        if close.get(field) is not True:
            return None, "LATE_QUOTE_PROVENANCE_UNVERIFIED"
    if not str(close.get("source_url") or "").startswith("https://"):
        return None, "LATE_QUOTE_SOURCE_URL_MISSING"
    try:
        origin, observed, start = (
            utc(close["source_quote_at"]), utc(close["observed_at"]), utc(row["kickoff"])
        )
        initial_observed = utc(row["observed_at"])
        if not initial_observed < origin <= observed < start:
            return None, "INVALID_LATE_QUOTE_TIMELINE"
        if observed - origin > MAX_ORIGIN_LAG or start - observed > LAST_QUOTE_WINDOW:
            return None, "LATE_QUOTE_TOO_OLD"
        if pub is None or not observed <= pub < start:
            return None, "LATE_QUOTE_NOT_PUBLISHED_PREGAME"
        entry_p = pair_probability(row["american_odds"], row["opposite_american_odds"])
        late_p = pair_probability(close["american_odds"], close["opposite_american_odds"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return None, "INVALID_LATE_QUOTE"
    return late_p - entry_p, "LATE_MARKET_SAME_LINE_NO_VIG_PROXY"


def evaluate_paper(root, *, commit_time=first_commit_time):
    root = Path(root)
    entries = _read_by_id(root / "entries")
    closes = _read_by_id(root / "late_quotes")
    finals = _read_by_id(root / "finals")
    used = set()
    rows = []
    for receipt_id, (entry, path) in sorted(entries.items()):
        pub = commit_time(path)
        issues = entry_reasons(entry, pub)
        duplicate_selection = (
            entry.get("sport"), entry.get("game_id"), entry.get("market"),
            entry.get("side"), str(entry.get("line")),
        )
        if duplicate_selection in used:
            issues.append("DUPLICATE_SELECTION_FOR_SAME_GAME")
        used.add(duplicate_selection)
        record = {"receipt_id": receipt_id, "status": "BLOCKED" if issues else "PENDING",
                  "blockers": sorted(set(issues)), "result": None, "profit_units": None,
                  "late_market_probability_change": None, "late_market_status": None}
        if issues:
            rows.append(record)
            continue
        close = closes.get(receipt_id)
        change, clv_status = _late_quote(
            entry, close[0] if close else None,
            commit_time(close[1]) if close else None,
        )
        record["late_market_probability_change"] = change
        record["late_market_status"] = clv_status
        final = finals.get(receipt_id)
        final_result = _result(entry, final[0]) if final else None
        if final_result is not None:
            record["result"], record["profit_units"] = final_result
            record["status"] = "SETTLED_PAPER"
        rows.append(record)
    settled = [r for r in rows if r["status"] == "SETTLED_PAPER"]
    economic = [r["profit_units"] for r in settled]
    proxies = [
        r["late_market_probability_change"] for r in rows
        if r["status"] == "SETTLED_PAPER"
        and r["late_market_probability_change"] is not None
    ]
    total = sum(economic)
    return {
        "spec": "point_in_time_paper_quote_v1",
        "status": "RESEARCH_ONLY_NO_BET_AUTHORIZATION",
        "paper_entries": len(entries),
        "blocked": sum(r["status"] == "BLOCKED" for r in rows),
        "pending": sum(r["status"] == "PENDING" for r in rows),
        "settled": len(settled),
        "wins": sum(r["result"] == "WIN" for r in settled),
        "losses": sum(r["result"] == "LOSS" for r in settled),
        "pushes": sum(r["result"] == "PUSH" for r in settled),
        "units": total,
        "roi_per_entry": total / len(economic) if economic else None,
        "late_market_samples": len(proxies),
        "average_late_market_probability_change": (
            sum(proxies) / len(proxies) if proxies else None
        ),
        "late_market_metric_is_certified_closing_line_value": False,
        "actual_wagers": 0,
        "betting_authorized": False,
        "model_promotion_authorized": False,
        "limitations": [
            "Human verification flags are attestations, not independent source verification",
            "Tracked git commit dates are insufficient proof of original book availability",
            "Late quote within 30 minutes is a proxy, not certified final close",
            "Missing late quotes are null, never zero and never imputed",
            "ROI covers vetted frozen one-unit hypothetical entries only",
            "No inference of profitable edge from zero or small prospective samples",
        ],
        "entries": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captures", type=Path, default=Path("history/price_scan_v1/captures")
    )
    parser.add_argument(
        "--paper-root", type=Path, default=Path("history/forward_verified_quotes_v1")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "source_audit": capture_audit(args.captures),
        "paper_replay": evaluate_paper(args.paper_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "distinct_source_records": report["source_audit"]["distinct_source_records"],
        "source_evidence_complete": report["source_audit"]["source_evidence_complete"],
        "prospective_paper_entries": report["paper_replay"]["paper_entries"],
        "settled_paper_entries": report["paper_replay"]["settled"],
        "verified_late_market_samples": report["paper_replay"]["late_market_samples"],
        "betting_authorized": False,
    }))


if __name__ == "__main__":
    main()
