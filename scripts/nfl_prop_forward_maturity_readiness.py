"""Read-only 2026 NFL frozen player-prop evaluation maturity calendar.

DOES NOT retrieve official results, sportsbook prices, game availability,
medical data, lineup status, bets, or futures. It measures only calendar
eligibility for later independent >=72-hour official-source verification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import argparse
import json
from pathlib import Path

from scripts.nfl_prop_paired_cdf_forward_grader import read_jsonl, utc
from scripts.nfl_prop_primary_anchor_score import anchor_results

VERSION = "NFL_PROP_FORWARD_MATURITY_READINESS_V1"
ANCHORS = Path("history/player_prop_primary_anchor_v1_2026.jsonl")
PAIRED = Path("history/player_prop_paired_cdf_lines_v1_2026.jsonl")
REPORT = Path("reports/nfl_prop_forward_maturity_readiness_2026.json")
MIN_SOURCE_MATURITY = timedelta(hours=72)
MIN_INDEPENDENT_GAMES_PER_MARKET = 20
FAMILYWISE_MARKETS = 4


def readiness(anchors, paired_originals, *, now):
    """Audit original contracts before deriving source-agnostic maturity windows.

    An elapsed 72-hour interval is ONLY a necessary time condition.
    It is not proof of final scores, provider completeness or player presence.
    """
    current = utc(now)
    # Existing primary-anchor validator tests full 7-line frozen CDF rows,
    # original SHA256, unique player-game-stat anchors, selection based on
    # ORIGINAL .50 tail (not outcomes), and pregame first-seen boundaries.
    integrity = anchor_results(anchors, paired_originals, [])
    if integrity["original_single_anchor_groups"] != len(anchors):
        raise ValueError("invalid registered first-seen anchor cohort")

    events = {}
    markets = defaultdict(lambda: Counter())
    player_market_seen = set()
    next_candidates = []
    for record in anchors:
        game_id = str(record["game_id"])
        player_id = str(record["player_id"])
        market = str(record["market"])
        kickoff = utc(record["kickoff_utc"])
        maturity = kickoff + MIN_SOURCE_MATURITY
        anchor_seen = utc(record["anchor_first_observed_at_utc"])
        if anchor_seen >= kickoff:
            raise ValueError("anchor original was not registered before kickoff")
        group=(game_id,player_id,market)
        if group in player_market_seen:
            raise ValueError("duplicate registered player-game-stat anchor")
        player_market_seen.add(group)

        previous = events.get(game_id)
        if previous is None:
            events[game_id] = {
                "game_id":game_id,
                "kickoff_utc":kickoff.isoformat(),
                "first_allowed_72h_source_check_utc":maturity.isoformat(),
                "preregistered_anchors":0,
                "by_market":{},
            }
        elif (utc(previous["kickoff_utc"]) != kickoff):
            raise ValueError("inconsistent original kickoff for same game")
        event = events[game_id]
        event["preregistered_anchors"] += 1
        event["by_market"][market] = event["by_market"].get(market,0)+1

        if current < kickoff:
            stage="ORIGINAL_GAME_HAS_NOT_KICKED_OFF"
        elif current < maturity:
            stage="BEFORE_REQUIRED_72H_SOURCE_MATURITY"
        else:
            stage="72H_ELAPSED_BUT_OFFICIAL_COMPLETION_AND_STATS_UNVERIFIED"
        event["stage"] = stage
        markets[market][stage] += 1
        if current < maturity:
            next_candidates.append((maturity,game_id))

    ordered=sorted(events.values(),key=lambda e:(e["kickoff_utc"],e["game_id"]))
    per_market={}
    for market in sorted(integrity["scoreboard"]):
        a=integrity["scoreboard"][market]
        n=sum(markets[market].values())
        if n != a["preregistered_original_player_game_market_anchors"]:
            raise ValueError("anchor counts do not match original source")
        # Time condition does not imply source eligibility, and no grades are
        # loaded in this job. Thus source-valid scoring counts remain unknown.
        per_market[market] = {
            "original_preregistered_anchors":n,
            "not_yet_kicked_off":markets[market]["ORIGINAL_GAME_HAS_NOT_KICKED_OFF"],
            "between_kickoff_and_72h":markets[market]["BEFORE_REQUIRED_72H_SOURCE_MATURITY"],
            "72h_elapsed_source_not_verified":markets[market][
                "72H_ELAPSED_BUT_OFFICIAL_COMPLETION_AND_STATS_UNVERIFIED"],
            "official_stat_source_eligible_anchors":None,
            "graded_from_official_stat_rows":None,
            "market_comparability_to_account_quotes":False,
        }

    time_elapsed=sum(e["preregistered_anchors"] for e in ordered
                     if e["stage"]=="72H_ELAPSED_BUT_OFFICIAL_COMPLETION_AND_STATS_UNVERIFIED")
    return {
        "version":VERSION,
        "as_of_utc":current.isoformat(),
        "source_data_loaded":"ORIGINAL_IMMUTABLE_PREGAME_JSONL_ONLY",
        "looked_up_postgame_scores":False,
        "looked_up_player_stats":False,
        "queried_bookmaker_prices":False,
        "min_72h_prospective_official_grade_gate_enforced":True,
        "original_paired_synthetic_thresholds":len(paired_originals),
        "original_primary_single_line_anchors":len(anchors),
        "unique_original_games":len(ordered),
        "markets":per_market,
        "game_readiness":ordered,
        "original_anchor_integrity_passed":True,
        "time_elapsed_anchor_count_not_same_as_source_graded_count":time_elapsed,
        "next_time_boundary_utc":min(next_candidates)[0].isoformat()
        if next_candidates else None,
        "earliest_time_window_opened_utc":(
            min((utc(e["first_allowed_72h_source_check_utc"]) for e in ordered)).isoformat()
            if ordered else None),
        "required_independent_source_games_per_market_for_CI":MIN_INDEPENDENT_GAMES_PER_MARKET,
        "multiplicity_adjusted_market_family":FAMILYWISE_MARKETS,
        "true_betmgm_ev":None,
        "validated_profitable_betmgm_edge":False,
        "account_accepted_betmgm_prices":0,
        "wagers":0,
        "decision":"PASS",
        "status":"RESEARCH_ONLY",
    }


def run(*, now=None, anchors_path=ANCHORS, paired_path=PAIRED, report_path=REPORT):
    at=utc(now or datetime.now(timezone.utc).isoformat())
    anchors=read_jsonl(anchors_path)
    paired=read_jsonl(paired_path)
    if not anchors or not paired:
        raise ValueError("no frozen first-seen 2026 anchor and paired-line source")
    report=readiness(anchors,paired,now=at.isoformat())
    path=Path(report_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
                    encoding="utf8")
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors",type=Path,default=ANCHORS)
    ap.add_argument("--paired",type=Path,default=PAIRED)
    ap.add_argument("--report",type=Path,default=REPORT)
    args=ap.parse_args()
    report=run(anchors_path=args.anchors,paired_path=args.paired,
               report_path=args.report)
    # Emit public-safe compact summary to CI (detailed per-game JSON artifact).
    print(json.dumps({k:report[k] for k in (
        "version","as_of_utc","original_primary_single_line_anchors",
        "original_paired_synthetic_thresholds","unique_original_games",
        "earliest_time_window_opened_utc","next_time_boundary_utc",
        "time_elapsed_anchor_count_not_same_as_source_graded_count",
        "markets","decision","true_betmgm_ev",
    )},sort_keys=True))


if __name__ == "__main__":
    main()
