"""Audit append-only NFL 2026 player-stat source grades against frozen originals.

The first grade and every later provider-stat correction keep independent
SHA-256s chained to the preceding grade for THAT exact player/game/stat/line.
This is research data-integrity, not independent proof of the external provider
or a settled sportsbook ticket. Never backfill before >=72h post kickoff.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math

from scripts.nfl_player_prop_full_distribution import _utc, hash_object

MATURE_HOURS = 72
ORIGINAL = "ORIGINAL_MATURE_GRADE"
CORRECTION = "APPENDED_OFFICIAL_SOURCE_STAT_CORRECTION"


def key(row):
    return (str(row["game_id"]), str(row["player_id"]),
            str(row["market"]), float(row["synthetic_line"]))


def record_hash(row):
    return hash_object({k:v for k,v in row.items()
                        if k != "grade_record_sha256"})


def stamp(row, prior=None):
    new = dict(row)
    new["previous_grade_sha256"] = (prior["grade_record_sha256"]
                                    if prior is not None else None)
    new["grade_record_sha256"] = record_hash(new)
    return new


def validate_grade_journal(grades, originals):
    """Fail closed on source-lineage, impossible outcome or revision edits.

    'Official' in source field names is a provider attribution. This check
    is not cryptographic certification that the provider's numbers are right.
    """
    frozen = {}
    for row in originals:
        k = key(row)
        if k in frozen:
            raise ValueError("duplicate original frozen prop contract")
        if row.get("frozen_row_sha256") != hash_object({
                field:value for field,value in row.items()
                if field != "frozen_row_sha256"}):
            raise ValueError("corrupt original frozen player forecast")
        if not _utc(row["first_observed_at_utc"]) < _utc(row["kickoff_utc"]):
            raise ValueError("original player forecast was not pregame")
        frozen[k] = row

    by_key = defaultdict(list)
    for g in grades:
        k = key(g)
        f = frozen.get(k)
        if f is None:
            raise ValueError("grade has no matching frozen original source")
        if g.get("grade_record_sha256") != record_hash(g):
            raise ValueError("tampered or unsigned grade journal record")
        if g.get("model") != f["model"]:
            raise ValueError("grade model lineage mismatch")
        if (g.get("original_row_sha256") != f["frozen_row_sha256"] or
                g.get("team") != f["team"] or
                _utc(g["kickoff_utc"]) != _utc(f["kickoff_utc"]) or
                _utc(g["original_first_seen_utc"]) != _utc(f["first_observed_at_utc"])):
            raise ValueError("grade replaced original player/game/source identity")
        if (g.get("p_original") != float(f["p_over_original_conditional_on_official_stat_row"])
                or g.get("p_snap") != float(f["p_over_snap_conditional_on_official_stat_row"])):
            raise ValueError("grade changed frozen pregame probabilities")
        when = _utc(g["grade_source_captured_utc"])
        if when < _utc(f["kickoff_utc"])+timedelta(hours=MATURE_HOURS):
            raise ValueError("grade entered before 72-hour source maturity")
        if (g.get("source_home_distinct_player_rows", 0) < 4 or
                g.get("source_away_distinct_player_rows", 0) < 4 or
                g["source_home_distinct_player_rows"]+
                g["source_away_distinct_player_rows"] < 12):
            raise ValueError("insufficient independent team/player stat source coverage")
        value = float(g["official_stat_value"])
        if not math.isfinite(value):
            raise ValueError("nonfinite source stat cannot be graded")
        if g["market"]=="receptions" and (value < 0 or not value.is_integer()):
            raise ValueError("invalid official reception count")
        y = g.get("target_over_given_official_stat_row")
        if isinstance(y, bool) or y not in (0, 1) or y != int(value > k[3]):
            raise ValueError("grade outcome conflicts with official stat and line")
        if (g.get("source_record_is_BetMGM_settlement") is not False or
                g.get("betting_decision")!="PASS" or
                g.get("actual_money_EV") is not None):
            raise ValueError("source stat outcome falsely promoted as book settlement")
        revision = g.get("revision")
        if type(revision) is not int or revision < 1:
            raise ValueError("invalid positive grade revision")
        by_key[k].append(g)

    for k, history in by_key.items():
        previous = None
        for index,g in enumerate(history,1):
            if g["revision"] != index:
                raise ValueError("grade correction revisions must be sequential and unique")
            if g.get("previous_grade_sha256") != (
                    previous["grade_record_sha256"] if previous is not None else None):
                raise ValueError("grade correction broken SHA predecessor chain")
            if index == 1:
                if (g.get("record_type") != ORIGINAL or
                        "previous_official_stat_value" in g):
                    raise ValueError("grade first revision must be original mature source")
            else:
                if (g.get("record_type") != CORRECTION or
                        g.get("previous_official_stat_value") != previous["official_stat_value"] or
                        g["official_stat_value"] == previous["official_stat_value"]):
                    raise ValueError("not a legitimate changed official source correction")
                if _utc(g["grade_source_captured_utc"]) < _utc(previous["grade_source_captured_utc"]):
                    raise ValueError("source correction captured before predecessor")
            previous = g
    return {k:history[-1] for k,history in by_key.items()}
