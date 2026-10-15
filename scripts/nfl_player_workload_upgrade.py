"""Generate and freeze the separately versioned, coherent workload forecast."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl.coupled_research_journal import archive
from nfl.player_workload_shrinkage import SPEC, history_index, upgrade

MODEL_FILES = (
    "nfl/coupled_drive_player_shadow.py",
    "nfl/coupled_nflverse_asof.py",
    "nfl/ratings.py",
    "nfl/data.py",
    "nfl/contracts.py",
    "scripts/nfl_coupled_drive_player_research_run.py",
    "nfl/player_workload_shrinkage.py",
    "scripts/nfl_player_workload_upgrade.py",
    "research/player_workload_shrinkage_frozen.json",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=Path("reports/coupled_drive_player_shadow.json"))
    p.add_argument("--replay", type=Path, default=Path("reports/coupled_replay_inputs.json"))
    p.add_argument("--stats", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/workload_v2_scan.json"))
    p.add_argument("--out-replay", type=Path, default=Path("reports/workload_v2_replay.json"))
    p.add_argument("--journal-root", type=Path)
    p.add_argument("--run-id")
    p.add_argument("--code-sha")
    a = p.parse_args()
    if a.journal_root and (not a.run_id or not a.code_sha):
        raise ValueError("JOURNAL_RUN_AND_CODE_REQUIRED")
    scan, replay = json.loads(a.scan.read_text()), json.loads(a.replay.read_text())
    frozen = json.loads(Path("research/player_workload_shrinkage_frozen.json").read_text())
    if frozen["spec"] != SPEC:
        raise ValueError("INVALID_FROZEN_WORKLOAD_FAMILY")
    if a.stats:
        stats = pl.read_parquet(a.stats)
    else:
        import nflreadpy as nfl

        stats = nfl.load_player_stats([scan["season"]])
    stats = stats.select(
        "season_type", "season", "week", "team", "player_id", "receptions", "carries"
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = a.out.with_suffix(".history.parquet")
    stats.write_parquet(snapshot)
    source_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    records = stats.to_dicts()
    valid_records = [r for r in records if r.get("player_id") and r.get("team")]
    excluded_identity_rows = len(records) - len(valid_records)
    if not valid_records:
        raise ValueError("NO_VALID_PLAYER_HISTORY")
    fingerprint = hashlib.sha256(
        b"".join(f.encode() + b"\0" + Path(f).read_bytes() for f in MODEL_FILES)
    ).hexdigest()
    captured = datetime.now(UTC)
    newscan, newreplay, audit = upgrade(
        scan,
        replay,
        history_index(valid_records),
        at=captured,
        settings=frozen["settings"],
        model_sha=fingerprint,
        source_sha=source_sha,
    )
    newscan["workload_source_quality"] = {
        "excluded_missing_player_or_team_rows": excluded_identity_rows,
        "missing_player_history_is_not_zero": True,
    }
    a.out.write_text(json.dumps(newscan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    a.out_replay.parent.mkdir(parents=True, exist_ok=True)
    a.out_replay.write_text(json.dumps(newreplay, indent=2, sort_keys=True, allow_nan=False) + "\n")
    audit_path = a.out.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(
            dict(
                spec=SPEC,
                captured_utc=captured.isoformat(),
                model_sha256=fingerprint,
                source_sha256=source_sha,
                source_publication_verified=False,
                source="https://github.com/nflverse/nflverse-data/releases/tag/player_stats",
                source_snapshot=str(snapshot),
                excluded_missing_player_or_team_rows=excluded_identity_rows,
                games=audit,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    receipt = None
    if a.journal_root:
        if a.journal_root == Path("history/coupled_primary_v1_2026"):
            raise ValueError("V2_REQUIRES_SEPARATE_COHORT")
        receipt = archive(
            a.journal_root,
            a.out,
            a.out_replay,
            at=datetime.now(UTC),
            run_id=a.run_id,
            code_sha=a.code_sha,
            model_sha=fingerprint,
            sidecars=[audit_path],
        )
        a.out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                spec=SPEC,
                games=len(newscan["games"]),
                model_sha256=fingerprint,
                receipt=receipt,
                betting_authorized=False,
            )
        )
    )


if __name__ == "__main__":
    main()
