"""Publish our own score, fair spread/total and player-stat forecast bundle."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from nfl.independent_publication import build_bundle, write_bundle
from nfl.public_context_inputs import unique_player_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=Path("reports/coupled_drive_player_shadow.json"))
    p.add_argument("--replay", type=Path, default=Path("reports/coupled_replay_inputs.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/independent_forecasts.json"))
    p.add_argument("--resolve-names", action="store_true")
    a = p.parse_args()
    names, name_errors, espn_ids = {}, [], {}
    identity_source = {"source": "nflreadpy.load_players", "status": "UNAVAILABLE"}
    if a.resolve_names:
        try:
            import nflreadpy as nfl

            rows = nfl.load_players().to_dicts()
            espn_ids = unique_player_ids(rows)
            identity_source.update(
                status="ONE_TO_ONE_SOURCE_DIRECTORY_IDS",
                captured_utc=datetime.now(UTC).isoformat(),
                mapping_sha256=hashlib.sha256(
                    json.dumps(espn_ids, sort_keys=True).encode()
                ).hexdigest(),
                unique_mappings=len(espn_ids),
            )
            for r in rows:
                pid = r.get("gsis_id")
                name = r.get("display_name") or r.get("football_name") or r.get("full_name")
                if pid and name:
                    names[str(pid)] = str(name)
            if not names:
                raise ValueError("NO_SOURCE_PLAYER_NAMES_AVAILABLE")
        except Exception as exc:
            # Name lookup cannot change numbers or impersonate verified actives.
            name_errors.append("PLAYER_NAME_SOURCE_FAILED:" + type(exc).__name__)
    bundle = build_bundle(
        json.loads(a.scan.read_text()),
        json.loads(a.replay.read_text()),
        at=datetime.now(UTC),
        names=names,
    )
    bundle["source_errors"] += name_errors
    bundle["player_identity_crosswalk"] = identity_source
    for player in bundle["player_props"]:
        # Supplemental source identity only; never infer active status from a name.
        player["espn_athlete_id"] = espn_ids.get(player["player_id"])
    bundle["source_scan_sha256"] = hashlib.sha256(a.scan.read_bytes()).hexdigest()
    bundle["source_replay_sha256"] = hashlib.sha256(a.replay.read_bytes()).hexdigest()
    outputs = write_bundle(bundle, a.out)
    print(
        json.dumps(
            dict(
                games_requested=bundle["games_requested"],
                games_forecast=bundle["games_forecast"],
                named_players=bundle["named_player_rows"],
                prop_metric_forecasts=bundle["prop_metric_forecasts"],
                accuracy_vs_market=bundle["accuracy_vs_market"],
                outputs=outputs,
                source_errors=bundle["source_errors"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
