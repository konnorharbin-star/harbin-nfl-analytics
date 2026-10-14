"""Regular-season 2025/2026 overtime scoring challenger, never a betting gate.

Possession-level abstraction of Rule 16. Inputs describe legally completed downs
and possession changes; penalties, unusual kick recoveries and double changes of
possession must be resolved upstream. This is not a play-by-play rules emulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Possession:
    outcome: str
    seconds: int
    try_points: int = 0

    def __post_init__(self):
        if self.outcome not in {"empty", "field_goal", "touchdown", "defensive_td", "safety"}:
            raise ValueError("unsupported possession outcome")
        if type(self.seconds) is not int or not 1 <= self.seconds <= 900:
            raise ValueError("duration must be integer seconds in [1, 900]")
        if type(self.try_points) is not int or self.try_points not in (0, 1, 2):
            raise ValueError("try points must be 0, 1 or 2")
        if self.outcome != "touchdown" and self.try_points:
            raise ValueError("only an offensive touchdown can carry try points")


def resolve_overtime(events, *, first_team=0):
    """Return incremental home/away points under the ten-minute regular-season cap.

    An event exceeding the remaining clock never awards its future score. Tries
    are untimed and taken only when they can affect the result. Input exhaustion
    before the clock expires is missing information, not a fabricated tie.
    """
    if type(first_team) is not int or first_team not in (0, 1):
        raise ValueError("first_team must be 0 (home) or 1 (away)")
    points = [0, 0]
    remaining = 600
    team = first_team
    opportunities = {first_team}  # Opening receiving team has its opportunity.
    sudden_death = False
    consumed = 0
    reason = None
    for event in events:
        if not isinstance(event, Possession):
            raise ValueError("events must be validated Possession objects")
        consumed += 1
        if event.seconds > remaining:
            remaining = 0
            reason = "CLOCK_EXPIRED_BEFORE_EVENT"
            break
        remaining -= event.seconds
        other = 1 - team
        opportunities.add(team)
        if event.outcome in ("defensive_td", "safety"):
            # Recovery gives the defense actual possession; opening safety is
            # an explicit Rule 16 exception. No unnecessary winning TD try.
            points[other] += 6 if event.outcome == "defensive_td" else 2
            reason = "DEFENSIVE_SCORE"
            break
        if event.outcome == "field_goal":
            points[team] += 3
        elif event.outcome == "touchdown":
            points[team] += 6
            wins_without_try = (
                sudden_death or len(opportunities) == 2 or remaining == 0
            ) and points[team] > points[other]
            if not wins_without_try:
                points[team] += event.try_points
        if sudden_death and event.outcome != "empty":
            reason = "SUDDEN_DEATH_SCORE"
            break
        if len(opportunities) == 2 and points[0] != points[1]:
            reason = "RESPONSE_COMPLETED"
            break
        if remaining == 0:
            reason = "CLOCK_EXPIRED"
            break
        if len(opportunities) == 2 or (consumed == 1 and event.outcome == "empty"):
            sudden_death = True
        team = other
    if reason is None:
        raise ValueError("incomplete overtime event stream")
    return {
        "home_points": points[0],
        "away_points": points[1],
        "seconds_remaining": remaining,
        "events_consumed": consumed,
        "reason": reason,
        "tie": points[0] == points[1],
    }


def append_overtime(regulation_scores, kernel, *, score_phase, seed=20261014):
    """Paired score-only sensitivity; rejects full-game inputs to prevent double OT.

    Kernel event frequencies/durations are empirical research assumptions, not
    calibrated team-specific OT probabilities. Player outcomes are not produced.
    """
    if score_phase != "VERIFIED_REGULATION_ONLY":
        raise ValueError("overtime requires verified regulation-only scores")
    scores = np.asarray(regulation_scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 2 or len(scores) == 0:
        raise ValueError("scores must be nonempty home/away pairs")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores != np.floor(scores)).any():
        raise ValueError("regulation scores must be finite nonnegative integers")
    if not kernel or any(not isinstance(e, Possession) for e in kernel):
        raise ValueError("a validated nonempty empirical possession kernel is required")
    final = scores.copy()
    rng = np.random.default_rng(seed)
    ties = np.flatnonzero(scores[:, 0] == scores[:, 1])
    for index in ties:
        # Minimum event duration is one second: 601 events guarantee a terminal
        # score or clock expiry without an arbitrary early tie fallback.
        events = [kernel[int(i)] for i in rng.integers(0, len(kernel), size=601)]
        result = resolve_overtime(events, first_team=int(rng.integers(0, 2)))
        final[index] += [result["home_points"], result["away_points"]]

    def describe(values):
        return {
            "home_points": float(values[:, 0].mean()),
            "away_points": float(values[:, 1].mean()),
            "home_spread": float(-(values[:, 0] - values[:, 1]).mean()),
            "total": float(values.sum(axis=1).mean()),
            "home_win": float((values[:, 0] > values[:, 1]).mean()),
            "away_win": float((values[:, 0] < values[:, 1]).mean()),
            "tie": float((values[:, 0] == values[:, 1]).mean()),
        }

    result = {
        "regulation": describe(scores),
        "with_overtime": describe(final),
        "draws": len(scores),
        "overtime_draws": len(ties),
        "score_phase": score_phase,
        "seed": seed,
        "status": "UNCALIBRATED_SCORE_ONLY_OT_CHALLENGER",
        "used_for_bet_ranking": False,
        "player_props_updated": False,
        "market_accuracy_proven": False,
        "promotion_earned": False,
    }
    if not math.isclose(
        sum(result["with_overtime"][k] for k in ("home_win", "away_win", "tie")), 1
    ):
        raise RuntimeError("invalid probability mass")
    return result
