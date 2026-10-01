from __future__ import annotations

import argparse
import json

from nfl.current import run_current_projection
from nfl.data import NFLDataClient
from nfl.recent_form_current import attach_current_recent_form_shadow
from nfl.recent_form_shadow import FROZEN_SELECTION_SEASONS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen recent-form totals shadow on the current NFL slate"
    )
    parser.add_argument("season", type=int, nargs="?", default=2026)
    parser.add_argument("--week", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source = NFLDataClient()
    canonical, canonical_audit = run_current_projection(
        args.season,
        args.week,
        client=source,
        refresh=args.refresh,
    )
    week = canonical_audit.week
    schedules = source.load_schedules(
        list(range(min(FROZEN_SELECTION_SEASONS) - 1, args.season + 1)),
        refresh=args.refresh,
    )
    pbp = source.load_pbp(
        sorted({*FROZEN_SELECTION_SEASONS, args.season}),
        refresh=args.refresh,
    )
    shadowed, shadow_audit = attach_current_recent_form_shadow(
        canonical,
        schedules,
        pbp,
        season=args.season,
        week=week,
    )
    baseline_unchanged = canonical.get_column("baseline_total").equals(
        shadowed.get_column("baseline_total")
    )
    payload = {
        "season": args.season,
        "week": week,
        "canonical_games": canonical.height,
        "baseline_total_unchanged": baseline_unchanged,
        "recent_form": shadow_audit.to_dict(),
        "shadow_total_min": float(shadowed.get_column("recent_form_shadow_total").min()),
        "shadow_total_max": float(shadowed.get_column("recent_form_shadow_total").max()),
        "meaning": (
            "Current recent-form totals are SHADOW diagnostics only; canonical totals "
            "remain unchanged and continue to drive market probabilities."
        ),
    }
    if not baseline_unchanged:
        raise RuntimeError("recent-form attachment changed the canonical total")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
