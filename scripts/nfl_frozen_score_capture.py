"""Append-only pre-kickoff score forecast snapshots (no model changes).

Run immediately after a model output is published, not retrospectively.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
from datetime import UTC, datetime
from pathlib import Path

FIELDS = [
    "game_id", "kickoff", "captured_at", "model_margin_home",
    "model_total", "model_home_win_probability", "source_sha256",
    "away_team", "home_team",
]


def parse_time(text: str) -> datetime:
    result = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp requires timezone")
    return result.astimezone(UTC)


def capture(source: bytes, existing: list[dict], now: datetime) -> tuple[list[dict], dict]:
    if now.tzinfo is None:
        raise ValueError("Capture time must be timezone-aware")
    stamp = now.astimezone(UTC)
    fingerprint = hashlib.sha256(source).hexdigest()
    incoming = list(csv.DictReader(io.StringIO(source.decode("utf-8-sig"))))
    seen = {(r["game_id"], r["source_sha256"]) for r in existing}
    new_rows: list[dict] = []
    rejected: dict[str, int] = {}
    ids_in_source = set()

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for r in incoming:
        game = str(r.get("game_id", "")).strip()
        if not game or game in ids_in_source:
            reject("missing_or_duplicate_game")
            continue
        ids_in_source.add(game)
        try:
            kickoff = parse_time(r["kickoff"])
            margin = float(r["model_margin_home"])
            total = float(r["model_total"])
            if not (math.isfinite(margin) and math.isfinite(total)):
                raise ValueError("non-finite")
            if total < abs(margin) or total < 0:
                raise ValueError("invalid score space")
        except (KeyError, TypeError, ValueError, OverflowError):
            reject("invalid_forecast")
            continue
        if stamp >= kickoff:
            reject("already_started")
            continue
        if (game, fingerprint) in seen:
            reject("already_captured_same_source")
            continue
        p = r.get("calibrated_home_probability", "")
        if p not in ("", None):
            try:
                pv = float(p)
                if not math.isfinite(pv) or pv < 0 or pv > 1:
                    raise ValueError("bad probability")
                p = str(pv)
            except (TypeError, ValueError):
                p = ""
        new_rows.append({
            "game_id": game,
            "kickoff": kickoff.isoformat(),
            "captured_at": stamp.isoformat(),
            "model_margin_home": str(margin),
            "model_total": str(total),
            "model_home_win_probability": p,
            "source_sha256": fingerprint,
            "away_team": r.get("away_team", ""),
            "home_team": r.get("home_team", ""),
        })
        seen.add((game, fingerprint))
    return existing + new_rows, {"input_games": len(incoming), "appended": len(new_rows), "rejected": rejected}


def run(source_path: Path, output_path: Path) -> dict:
    source = source_path.read_bytes()
    existing = []
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing ledger schema differs; refusing overwrite")
            existing = list(reader)
    rows, report = capture(source, existing, datetime.now(UTC))
    if report["appended"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(output_path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(output_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/suggested_bets.csv")
    parser.add_argument("--ledger", default="outputs/frozen_score_forecasts.csv")
    a = parser.parse_args()
    print(run(Path(a.source), Path(a.ledger)))


if __name__ == "__main__":
    main()
