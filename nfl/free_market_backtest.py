"""NCAA-style free historical market backtest for the NFL model.

The football projection is rebuilt week by week with no sportsbook inputs. Only after
that projection exists do free nflverse archive prices enter for probability/EV
comparison and grading. Opening-line observations are used when the free nflverse
initial-line file contains them; otherwise the archive-final value is an explicit
fallback, matching the CFB archive philosophy without pretending it is a timestamped
opening quote.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import polars as pl

from .contracts import DataContractError, require_columns
from .data import completed_games
from .free_market import FreeArchiveQuote, FreeNFLMarketStore
from .market import MarketComparison, MarketQuote, compare_two_way_market, remove_two_way_vig
from .probability import GaussianScoreDistribution
from .probability_runtime import build_operational_probability_distribution
from .ratings import fit_pregame_fair_score

PROJECTION_REQUIRED = {
    "season",
    "week",
    "game_id",
    "home_team",
    "away_team",
    "projected_home_margin",
    "projected_total",
    "actual_home_margin",
    "actual_total",
}
BET_REQUIRED = {
    "season",
    "week",
    "game_id",
    "market_type",
    "side",
    "probability_edge",
    "expected_value_per_unit",
    "result",
    "net_units",
}


@dataclass(frozen=True)
class ArchiveBetSummary:
    bets: int
    wins: int
    losses: int
    pushes: int
    net_units: float
    roi_per_unit_staked: float
    max_drawdown: float
    average_probability_edge: float
    average_expected_value_per_unit: float
    average_clv_proxy: float | None
    clv_samples: int
    roi_ci_95_low: float | None
    roi_ci_95_high: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveRule:
    min_probability_edge: float
    min_expected_value_per_unit: float


@dataclass(frozen=True)
class ArchiveHoldoutEvaluation:
    market_type: str
    validation_season: int
    holdout_season: int
    rule: ArchiveRule | None
    validation: ArchiveBetSummary
    holdout: ArchiveBetSummary
    candidate_pass: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["rule"] = None if self.rule is None else asdict(self.rule)
        result["validation"] = asdict(self.validation)
        result["holdout"] = asdict(self.holdout)
        return result


def build_archive_projection_dataset(
    schedules: pl.DataFrame,
    *,
    start_season: int,
    end_season: int,
) -> pl.DataFrame:
    """Rebuild the canonical fair score independently before every historical week."""

    if start_season > end_season:
        raise ValueError("start_season must be <= end_season")
    games = completed_games(schedules).filter(pl.col("game_type") == "REG")
    rows: list[dict[str, object]] = []
    for season in range(start_season, end_season + 1):
        season_games = games.filter(pl.col("season") == season)
        if season_games.is_empty():
            continue
        weeks = sorted(int(value) for value in season_games.get_column("week").unique())
        for week in weeks:
            targets = season_games.filter(pl.col("week") == week).sort("game_id")
            if targets.is_empty():
                continue
            model = fit_pregame_fair_score(schedules, season, week)
            for game in targets.iter_rows(named=True):
                home_team = str(game["home_team"])
                away_team = str(game["away_team"])
                projection = model.project(home_team, away_team)
                home_score = float(game["home_score"])
                away_score = float(game["away_score"])
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_id": str(game["game_id"]),
                        "gameday": game["gameday"],
                        "home_team": home_team,
                        "away_team": away_team,
                        "projected_home_margin": projection.home_margin,
                        "projected_total": projection.total,
                        "actual_home_margin": home_score - away_score,
                        "actual_total": home_score + away_score,
                    }
                )
    if not rows:
        raise DataContractError("free archive projection dataset is empty")
    return pl.DataFrame(rows).sort(["season", "week", "game_id"])


def _grade(
    market_type: str,
    side: str,
    line: float | None,
    actual_margin: float,
    actual_total: float,
    decimal_odds: float,
) -> tuple[str, float]:
    if market_type == "moneyline":
        value = actual_margin if side == "home" else -actual_margin
    elif market_type == "spread":
        if line is None:
            raise DataContractError("spread grading requires a line")
        value = actual_margin + line if side == "home" else -actual_margin + line
    elif market_type == "total":
        if line is None:
            raise DataContractError("total grading requires a line")
        value = actual_total - line if side == "over" else line - actual_total
    else:
        raise DataContractError(f"unsupported archive market type: {market_type}")

    if value > 1e-12:
        return "win", decimal_odds - 1.0
    if value < -1e-12:
        return "loss", -1.0
    return "push", 0.0


def _moneyline_clv(side: str, quote: FreeArchiveQuote) -> float | None:
    if not quote.has_distinct_open_moneyline:
        return None
    values = (
        quote.open_home_ml,
        quote.open_away_ml,
        quote.final_home_ml,
        quote.final_away_ml,
    )
    if any(value is None for value in values):
        return None
    assert quote.open_home_ml is not None and quote.open_away_ml is not None
    assert quote.final_home_ml is not None and quote.final_away_ml is not None
    open_home, open_away = remove_two_way_vig(quote.open_home_ml, quote.open_away_ml)
    final_home, final_away = remove_two_way_vig(quote.final_home_ml, quote.final_away_ml)
    return final_home - open_home if side == "home" else final_away - open_away


def _line_clv(
    market_type: str,
    side: str,
    decision_line: float | None,
    quote: FreeArchiveQuote,
) -> float | None:
    if decision_line is None:
        return None
    if market_type == "spread":
        if not quote.has_distinct_open_spread or quote.final_home_spread is None:
            return None
        closing = (
            quote.final_home_spread if side == "home" else -quote.final_home_spread
        )
        return decision_line - closing
    if market_type == "total":
        if not quote.has_distinct_open_total or quote.final_total is None:
            return None
        if side == "over":
            return quote.final_total - decision_line
        return decision_line - quote.final_total
    if market_type == "moneyline":
        return _moneyline_clv(side, quote)
    return None


def _pair_comparisons(
    distribution: GaussianScoreDistribution,
    projection: dict[str, object],
    quote: FreeArchiveQuote,
) -> list[MarketComparison]:
    output: list[MarketComparison] = []
    margin = float(projection["projected_home_margin"])
    total = float(projection["projected_total"])

    if quote.open_home_ml is not None and quote.open_away_ml is not None:
        output.extend(
            compare_two_way_market(
                distribution,
                projected_home_margin=margin,
                projected_total=total,
                first=MarketQuote(
                    market_type="moneyline",
                    side="home",
                    american_odds=quote.open_home_ml,
                    book=quote.archive_book,
                ),
                second=MarketQuote(
                    market_type="moneyline",
                    side="away",
                    american_odds=quote.open_away_ml,
                    book=quote.archive_book,
                ),
            )
        )

    if (
        quote.open_home_spread is not None
        and quote.open_home_spread_odds is not None
        and quote.open_away_spread_odds is not None
    ):
        output.extend(
            compare_two_way_market(
                distribution,
                projected_home_margin=margin,
                projected_total=total,
                first=MarketQuote(
                    market_type="spread",
                    side="home",
                    line=quote.open_home_spread,
                    american_odds=quote.open_home_spread_odds,
                    book=quote.archive_book,
                ),
                second=MarketQuote(
                    market_type="spread",
                    side="away",
                    line=-quote.open_home_spread,
                    american_odds=quote.open_away_spread_odds,
                    book=quote.archive_book,
                ),
            )
        )

    if (
        quote.open_total is not None
        and quote.open_over_odds is not None
        and quote.open_under_odds is not None
    ):
        output.extend(
            compare_two_way_market(
                distribution,
                projected_home_margin=margin,
                projected_total=total,
                first=MarketQuote(
                    market_type="total",
                    side="over",
                    line=quote.open_total,
                    american_odds=quote.open_over_odds,
                    book=quote.archive_book,
                ),
                second=MarketQuote(
                    market_type="total",
                    side="under",
                    line=quote.open_total,
                    american_odds=quote.open_under_odds,
                    book=quote.archive_book,
                ),
            )
        )
    return output


def build_free_archive_bets(
    projections: pl.DataFrame,
    market_store: FreeNFLMarketStore,
    *,
    min_probability_training_games: int = 64,
    enforce_probability_reliability: bool = False,
) -> pl.DataFrame:
    """Create one best-side opportunity per game/market with chronological probabilities."""

    require_columns(projections, PROJECTION_REQUIRED, "free_archive_projections")
    if min_probability_training_games < 64:
        raise ValueError("min_probability_training_games must be >= 64")

    rows: list[dict[str, object]] = []
    groups = projections.select("season", "week").unique().sort(["season", "week"])
    for group in groups.iter_rows(named=True):
        season = int(group["season"])
        week = int(group["week"])
        history = projections.filter(
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        if history.height < min_probability_training_games:
            continue
        distribution, probability_meta = build_operational_probability_distribution(
            history,
            current_season=season,
        )
        if (
            enforce_probability_reliability
            and not bool(probability_meta.get("reliability_ready", False))
        ):
            continue
        targets = projections.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        ).sort("game_id")
        for projection in targets.iter_rows(named=True):
            quote = market_store.quote(str(projection["game_id"]))
            if quote is None:
                continue
            comparisons = _pair_comparisons(distribution, projection, quote)
            by_market: dict[str, list[MarketComparison]] = {}
            for comparison in comparisons:
                by_market.setdefault(comparison.market_type, []).append(comparison)
            for market_type, candidates in sorted(by_market.items()):
                candidates.sort(
                    key=lambda item: (
                        item.expected_value_per_unit,
                        item.probability_edge,
                        item.american_odds,
                        item.side,
                    ),
                    reverse=True,
                )
                chosen = candidates[0]
                result, net_units = _grade(
                    chosen.market_type,
                    chosen.side,
                    chosen.line,
                    float(projection["actual_home_margin"]),
                    float(projection["actual_total"]),
                    chosen.decimal_odds,
                )
                if market_type == "moneyline":
                    distinct_open = quote.has_distinct_open_moneyline
                elif market_type == "spread":
                    distinct_open = quote.has_distinct_open_spread
                else:
                    distinct_open = quote.has_distinct_open_total
                rows.append(
                    {
                        **chosen.to_dict(),
                        "season": season,
                        "week": week,
                        "game_id": str(projection["game_id"]),
                        "home_team": str(projection["home_team"]),
                        "away_team": str(projection["away_team"]),
                        "provider": quote.provider,
                        "opening_book": quote.opening_book,
                        "archive_book": quote.archive_book,
                        "has_distinct_open": distinct_open,
                        "price_stage": (
                            "archive_open_line_final_price"
                            if distinct_open
                            else "archive_final_fallback"
                        ),
                        "clv_proxy": _line_clv(
                            chosen.market_type,
                            chosen.side,
                            chosen.line,
                            quote,
                        ),
                        "result": result,
                        "net_units": net_units,
                        "probability_model_family": probability_meta.get("model_family"),
                        "probability_reliability_ready": bool(
                            probability_meta.get("reliability_ready", False)
                        ),
                    }
                )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["season", "week", "game_id", "market_type"])


def _max_drawdown(profits: np.ndarray) -> float:
    if profits.size == 0:
        return 0.0
    curve = np.cumsum(profits)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], curve)))[:-1]
    return float(np.max(running_peak - curve))


def _roi_ci(
    profits: np.ndarray,
    *,
    seed: int = 26,
    samples: int = 3000,
) -> tuple[float | None, float | None]:
    if profits.size < 30:
        return None, None
    rng = np.random.default_rng(seed)
    roi_samples = np.empty(samples, dtype=float)
    for index in range(samples):
        roi_samples[index] = float(rng.choice(profits, size=profits.size, replace=True).mean())
    low, high = np.quantile(roi_samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_archive_bets(
    frame: pl.DataFrame,
    *,
    min_probability_edge: float = 0.0,
    min_expected_value_per_unit: float = 0.0,
) -> ArchiveBetSummary:
    require_columns(frame, BET_REQUIRED, "free_archive_bets")
    selected = frame.filter(
        (pl.col("probability_edge") >= min_probability_edge)
        & (pl.col("expected_value_per_unit") >= min_expected_value_per_unit)
    )
    if selected.is_empty():
        return ArchiveBetSummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0, None, None)

    profits = np.asarray(selected.get_column("net_units"), dtype=float)
    if "clv_proxy" in selected.columns:
        clv = selected.get_column("clv_proxy").drop_nulls()
    else:
        clv = pl.Series([], dtype=pl.Float64)
    low, high = _roi_ci(profits)
    net_units = float(profits.sum())
    return ArchiveBetSummary(
        bets=selected.height,
        wins=selected.filter(pl.col("result") == "win").height,
        losses=selected.filter(pl.col("result") == "loss").height,
        pushes=selected.filter(pl.col("result") == "push").height,
        net_units=net_units,
        roi_per_unit_staked=net_units / selected.height,
        max_drawdown=_max_drawdown(profits),
        average_probability_edge=float(selected.get_column("probability_edge").mean()),
        average_expected_value_per_unit=float(
            selected.get_column("expected_value_per_unit").mean()
        ),
        average_clv_proxy=None if clv.len() == 0 else float(clv.mean()),
        clv_samples=clv.len(),
        roi_ci_95_low=low,
        roi_ci_95_high=high,
    )


def evaluate_archive_holdout(
    bets: pl.DataFrame,
    *,
    market_type: str,
    validation_season: int,
    holdout_season: int,
    probability_edge_grid: tuple[float, ...] = (0.0, 0.02, 0.03, 0.04, 0.05),
    expected_value_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04),
    min_validation_bets: int = 25,
    min_holdout_bets: int = 25,
) -> ArchiveHoldoutEvaluation:
    """Select an archive rule on validation only, then score a later season once."""

    require_columns(bets, BET_REQUIRED, "free_archive_bets")
    if validation_season >= holdout_season:
        raise ValueError("validation_season must be earlier than holdout_season")
    if market_type not in {"moneyline", "spread", "total"}:
        raise ValueError(f"unsupported market_type: {market_type}")

    market = bets.filter(pl.col("market_type") == market_type)
    validation = market.filter(pl.col("season") == validation_season)
    holdout = market.filter(pl.col("season") == holdout_season)
    if validation.is_empty() or holdout.is_empty():
        raise DataContractError("validation and holdout seasons must both contain archive bets")

    candidates: list[tuple[tuple[float, ...], ArchiveRule, ArchiveBetSummary]] = []
    for edge in probability_edge_grid:
        for ev in expected_value_grid:
            rule = ArchiveRule(float(edge), float(ev))
            summary = summarize_archive_bets(
                validation,
                min_probability_edge=rule.min_probability_edge,
                min_expected_value_per_unit=rule.min_expected_value_per_unit,
            )
            if summary.bets < min_validation_bets:
                continue
            clv_score = summary.average_clv_proxy or 0.0
            score = (
                summary.net_units,
                summary.roi_per_unit_staked,
                clv_score,
                -rule.min_probability_edge,
                -rule.min_expected_value_per_unit,
            )
            candidates.append((score, rule, summary))

    empty = summarize_archive_bets(validation.head(0))
    if not candidates:
        return ArchiveHoldoutEvaluation(
            market_type,
            validation_season,
            holdout_season,
            None,
            empty,
            empty,
            False,
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, rule, validation_summary = candidates[0]
    if validation_summary.net_units <= 0:
        return ArchiveHoldoutEvaluation(
            market_type,
            validation_season,
            holdout_season,
            None,
            validation_summary,
            empty,
            False,
        )

    holdout_summary = summarize_archive_bets(
        holdout,
        min_probability_edge=rule.min_probability_edge,
        min_expected_value_per_unit=rule.min_expected_value_per_unit,
    )
    clv_gate = (
        holdout_summary.clv_samples == 0
        or (
            holdout_summary.average_clv_proxy is not None
            and isfinite(holdout_summary.average_clv_proxy)
            and holdout_summary.average_clv_proxy > 0
        )
    )
    candidate_pass = (
        holdout_summary.bets >= min_holdout_bets
        and holdout_summary.net_units > 0
        and holdout_summary.roi_per_unit_staked > 0
        and clv_gate
    )
    return ArchiveHoldoutEvaluation(
        market_type,
        validation_season,
        holdout_season,
        rule,
        validation_summary,
        holdout_summary,
        candidate_pass,
    )
