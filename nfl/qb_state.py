"""Leakage-safe quarterback state derived from historical player statistics.

This layer deliberately does not use the eventual quarterback from the target game.
For each team/week it uses the primary quarterback from that team's most recent
completed regular-season game as a *last-observed QB proxy*. That proxy and all QB
performance statistics are therefore known before the target week. Sportsbook prices,
current injury reports, and target-game participation are not inputs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

import polars as pl

from .contracts import PLAYER_STATS_REQUIRED, DataContractError, require_columns

QB_PRIOR_DROPBACKS = 75.0
QB_STATE_METRICS = (
    "epa",
    "cpoe",
    "sack_rate",
    "int_rate",
)


@dataclass
class _Aggregate:
    dropbacks: float = 0.0
    attempts: float = 0.0
    sacks: float = 0.0
    interceptions: float = 0.0
    passing_epa: float = 0.0
    cpoe_attempt_sum: float = 0.0
    cpoe_attempts: float = 0.0

    def add(self, row: dict[str, object]) -> None:
        attempts = _number(row.get("attempts"))
        sacks = _number(row.get("sacks_suffered"))
        interceptions = _number(row.get("passing_interceptions"))
        passing_epa = _number(row.get("passing_epa"))
        dropbacks = attempts + sacks

        self.dropbacks += dropbacks
        self.attempts += attempts
        self.sacks += sacks
        self.interceptions += interceptions
        self.passing_epa += passing_epa

        cpoe = row.get("passing_cpoe")
        if cpoe is not None:
            value = float(cpoe)
            if isfinite(value) and attempts > 0:
                self.cpoe_attempt_sum += value * attempts
                self.cpoe_attempts += attempts


def _number(value: object) -> float:
    if value is None:
        return 0.0
    out = float(value)
    return out if isfinite(out) else 0.0


def pregame_qb_history(player_stats: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Return same-season QB rows known strictly before ``week``."""

    require_columns(player_stats, PLAYER_STATS_REQUIRED, "player_stats")
    if week < 2:
        raise DataContractError("QB state requires target week >= 2")
    return player_stats.filter(
        (pl.col("season") == season)
        & (pl.col("week") < week)
        & (pl.col("season_type") == "REG")
        & (pl.col("position") == "QB")
        & pl.col("player_id").is_not_null()
        & pl.col("team").is_not_null()
        & pl.col("game_id").is_not_null()
    )


def primary_qb_games(history: pl.DataFrame) -> pl.DataFrame:
    """Choose the largest-dropback QB for each completed team-game."""

    require_columns(history, PLAYER_STATS_REQUIRED, "qb_history")
    best: dict[tuple[str, str], tuple[tuple[float, float, str], dict[str, object]]] = {}

    for row in history.iter_rows(named=True):
        team = str(row["team"])
        game_id = str(row["game_id"])
        attempts = _number(row.get("attempts"))
        sacks = _number(row.get("sacks_suffered"))
        dropbacks = attempts + sacks
        if dropbacks <= 0:
            continue
        player_id = str(row["player_id"])
        # Maximize dropbacks, then attempts; deterministic player-ID tie break.
        rank = (dropbacks, attempts, "".join(chr(255 - ord(c)) for c in player_id))
        key = (team, game_id)
        if key not in best or rank > best[key][0]:
            selected = dict(row)
            selected["dropbacks"] = dropbacks
            best[key] = (rank, selected)

    if not best:
        raise DataContractError("no primary QB game observations are available")

    rows = [value[1] for value in best.values()]
    return pl.DataFrame(rows).select(
        "season",
        "week",
        "game_id",
        "team",
        "player_id",
        "player_name",
        "dropbacks",
    ).sort(["team", "week", "game_id"])


def _shrunk_rates(
    aggregate: _Aggregate,
    league: _Aggregate,
    prior_dropbacks: float,
) -> dict[str, float]:
    if league.dropbacks <= 0 or league.attempts <= 0:
        raise DataContractError("league QB history is too small to establish priors")

    league_epa = league.passing_epa / league.dropbacks
    league_sack = league.sacks / league.dropbacks
    league_int = league.interceptions / league.attempts
    league_cpoe = (
        league.cpoe_attempt_sum / league.cpoe_attempts
        if league.cpoe_attempts > 0
        else 0.0
    )

    epa = (
        aggregate.passing_epa + (prior_dropbacks * league_epa)
    ) / (aggregate.dropbacks + prior_dropbacks)
    sack_rate = (
        aggregate.sacks + (prior_dropbacks * league_sack)
    ) / (aggregate.dropbacks + prior_dropbacks)

    attempt_prior = prior_dropbacks * (league.attempts / league.dropbacks)
    int_rate = (
        aggregate.interceptions + (attempt_prior * league_int)
    ) / (aggregate.attempts + attempt_prior)
    cpoe = (
        aggregate.cpoe_attempt_sum + (attempt_prior * league_cpoe)
    ) / (aggregate.cpoe_attempts + attempt_prior)

    return {
        "epa": float(epa),
        "cpoe": float(cpoe),
        "sack_rate": float(sack_rate),
        "int_rate": float(int_rate),
        "league_epa": float(league_epa),
        "league_cpoe": float(league_cpoe),
        "league_sack_rate": float(league_sack),
        "league_int_rate": float(league_int),
    }


def team_qb_state(
    player_stats: pl.DataFrame,
    season: int,
    week: int,
    *,
    prior_dropbacks: float = QB_PRIOR_DROPBACKS,
) -> pl.DataFrame:
    """Build one last-observed quarterback state row per team.

    Player rates are shrunk toward the same pregame league history. Team-relative
    deltas compare the last-observed QB to all quarterback dropbacks taken by that
    team before the target week. Continuity uses only historical primary-QB games.
    """

    if prior_dropbacks <= 0:
        raise ValueError("prior_dropbacks must be > 0")
    history = pregame_qb_history(player_stats, season, week)
    if history.is_empty():
        raise DataContractError("no pregame QB history is available")

    primaries = primary_qb_games(history)
    by_player: dict[str, _Aggregate] = defaultdict(_Aggregate)
    by_team: dict[str, _Aggregate] = defaultdict(_Aggregate)
    league = _Aggregate()

    for row in history.iter_rows(named=True):
        attempts = _number(row.get("attempts"))
        sacks = _number(row.get("sacks_suffered"))
        if attempts + sacks <= 0:
            continue
        player_id = str(row["player_id"])
        team = str(row["team"])
        by_player[player_id].add(row)
        by_team[team].add(row)
        league.add(row)

    primary_by_team: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in primaries.iter_rows(named=True):
        primary_by_team[str(row["team"])].append(row)

    rows: list[dict[str, object]] = []
    for team, games in sorted(primary_by_team.items()):
        games.sort(key=lambda row: (int(row["week"]), str(row["game_id"])))
        latest = games[-1]
        qb_id = str(latest["player_id"])
        if qb_id not in by_player or team not in by_team:
            raise DataContractError(f"missing aggregate QB state for {team}")

        qb_rates = _shrunk_rates(by_player[qb_id], league, prior_dropbacks)
        team_rates = _shrunk_rates(by_team[team], league, prior_dropbacks)

        consecutive = 0
        for game in reversed(games):
            if str(game["player_id"]) != qb_id:
                break
            consecutive += 1
        changed = len(games) >= 2 and str(games[-2]["player_id"]) != qb_id
        starts = sum(1 for game in games if str(game["player_id"]) == qb_id)
        aggregate = by_player[qb_id]
        confidence = aggregate.dropbacks / (aggregate.dropbacks + prior_dropbacks)

        rows.append(
            {
                "team": team,
                "qb_proxy_id": qb_id,
                "qb_proxy_name": latest.get("player_name"),
                "qb_proxy_last_week": int(latest["week"]),
                "qb_dropbacks": float(aggregate.dropbacks),
                "qb_confidence": float(confidence),
                "qb_starts": int(starts),
                "qb_consecutive_starts": int(consecutive),
                "qb_changed_last_observation": int(changed),
                "qb_epa_per_dropback": qb_rates["epa"],
                "qb_cpoe": qb_rates["cpoe"],
                "qb_sack_rate": qb_rates["sack_rate"],
                "qb_int_rate": qb_rates["int_rate"],
                "qb_epa_vs_team": qb_rates["epa"] - team_rates["epa"],
                "qb_cpoe_vs_team": qb_rates["cpoe"] - team_rates["cpoe"],
                "qb_sack_rate_vs_team": team_rates["sack_rate"] - qb_rates["sack_rate"],
                "qb_int_rate_vs_team": team_rates["int_rate"] - qb_rates["int_rate"],
                "league_qb_epa_per_dropback": qb_rates["league_epa"],
                "league_qb_cpoe": qb_rates["league_cpoe"],
                "league_qb_sack_rate": qb_rates["league_sack_rate"],
                "league_qb_int_rate": qb_rates["league_int_rate"],
            }
        )

    if len(rows) < 2:
        raise DataContractError("QB state contains fewer than two teams")
    return pl.DataFrame(rows).sort("team")


def qb_matchup_signals(home: dict[str, object], away: dict[str, object]) -> dict[str, float]:
    """Create target-specific QB margin/total signals from pregame state rows."""

    values: dict[str, float] = {}
    specs = {
        "epa": ("qb_epa_per_dropback", "league_qb_epa_per_dropback", 1.0),
        "cpoe": ("qb_cpoe", "league_qb_cpoe", 1.0),
        "sack_rate": ("qb_sack_rate", "league_qb_sack_rate", -1.0),
        "int_rate": ("qb_int_rate", "league_qb_int_rate", -1.0),
    }
    for name, (column, league_column, direction) in specs.items():
        home_value = direction * float(home[column])
        away_value = direction * float(away[column])
        league_value = direction * float(home[league_column])
        values[f"qb_{name}_margin_signal"] = home_value - away_value
        values[f"qb_{name}_total_signal"] = home_value + away_value - (2.0 * league_value)

    delta_specs = {
        "delta_epa": "qb_epa_vs_team",
        "delta_cpoe": "qb_cpoe_vs_team",
        "delta_sack": "qb_sack_rate_vs_team",
        "delta_int": "qb_int_rate_vs_team",
    }
    for name, column in delta_specs.items():
        home_value = float(home[column])
        away_value = float(away[column])
        values[f"qb_{name}_margin_signal"] = home_value - away_value
        values[f"qb_{name}_total_signal"] = home_value + away_value

    home_continuity = float(home["qb_consecutive_starts"])
    away_continuity = float(away["qb_consecutive_starts"])
    home_change = float(home["qb_changed_last_observation"])
    away_change = float(away["qb_changed_last_observation"])
    home_confidence = float(home["qb_confidence"])
    away_confidence = float(away["qb_confidence"])

    values["qb_continuity_margin_signal"] = home_continuity - away_continuity
    values["qb_continuity_total_signal"] = home_continuity + away_continuity
    values["qb_change_margin_signal"] = away_change - home_change
    values["qb_change_total_signal"] = -(home_change + away_change)
    values["qb_confidence_margin_signal"] = home_confidence - away_confidence
    values["qb_confidence_total_signal"] = home_confidence + away_confidence
    return values
