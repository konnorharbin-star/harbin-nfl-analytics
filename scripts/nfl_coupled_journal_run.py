"""Persist and grade the new coupled NFL cohort; never place or approve bets."""
import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from nfl.coupled_research_journal import archive, grade_archive, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "grade", "check-pregame"))
    parser.add_argument("--root", type=Path, default=Path("history/coupled_primary_v1_2026"))
    parser.add_argument("--snapshot", type=Path, default=Path("reports/coupled_drive_player_shadow.json"))
    parser.add_argument("--replay", type=Path, default=Path("reports/coupled_replay_inputs.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--code-sha")
    parser.add_argument("--sidecar", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("reports/coupled_durable_status.json"))
    args = parser.parse_args()
    at = datetime.now(UTC)
    if args.mode == "check-pregame":
        from nfl.coupled_research_journal import read_journal
        from nfl.coupled_forward_grading import clock
        for row in read_journal(args.root / "primary.jsonl"):
            if row["source_run_id"] == args.run_id and at >= clock(row["kickoff_utc"]):
                raise ValueError("NEW_CAPTURE_COMMIT_WOULD_BE_POSTKICKOFF")
        return
    if args.mode == "freeze":
        files=("nfl/coupled_drive_player_shadow.py","nfl/coupled_nflverse_asof.py",
               "nfl/ratings.py","nfl/data.py","nfl/contracts.py",
               "scripts/nfl_coupled_drive_player_research_run.py")
        model_sha=digest(b"".join(name.encode()+b"\0"+Path(name).read_bytes() for name in files))
        result = archive(args.root, args.snapshot, args.replay, at=at,
                         run_id=args.run_id, code_sha=args.code_sha, sidecars=args.sidecar,
                         model_sha=model_sha)
    else:
        import nflreadpy as nfl
        result = grade_archive(args.root, nfl.load_schedules(2026).to_dicts(),
                               nfl.load_player_stats(2026).to_dicts(), at=at)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
