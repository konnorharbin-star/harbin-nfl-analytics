"""Free ESPN archived NFL opening/closing market evidence.

ESPN Core stores explicit provider-labeled open and close snapshots for completed
games. These are archived market stages, not point-in-time captures, so this module
keeps them separate from market_history and never fabricates timestamps.

Sportsbook prices remain strictly downstream of the chronological football model.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from .book_identity import canonical_book_identity
from .contracts import DataContractError, require_columns
from .espn_market import ESPN_CORE_ODDS_URL, ESPN_SCOREBOARD_URL, _event_teams
from .free_market_backtest import PROJECTION_REQUIRED, summarize_archive_bets
from .market import MarketQuote, compare_two_way_market, remove_two_way_vig
from .market_backtest import grade_market_comparisons
from .probability_runtime import build_operational_probability_distribution

ARCHIVE_QUOTE_REQUIRED = {
    "game_id",
    "market_type",
    "side",
    "line",
    "american_odds",
    "provider",
    "book",
    "provider_id",
    "source_event_id",
    "snapshot_id",
    "archive_stage",
}


@dataclass(frozen=True)
class ESPNArchiveCoverage:
    requested_games: int
    matched_events: int
    games_with_open: int
    games_with_close: int
    entry_pairs: int
    closing_pairs: int
    books: tuple[str, ...]
    source_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _number(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if numeric != numeric else numeric


def _american(value: object) -> int | None:
    if isinstance(value, dict):
        value = value.get("american")
    numeric = _number(value)
    if numeric is None:
        return None
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        return None
    if rounded == 0 or -100 < rounded < 100:
        return None
    return rounded


def _line(value: object) -> float | None:
    if isinstance(value, dict):
        value = value.get("american")
    return _number(value)


def _provider(item: dict[str, object]) -> tuple[str, str]:
    provider = item.get("provider")
    if isinstance(provider, dict):
        provider_id = str(provider.get("id") or "").strip()
        provider_name = str(
            provider.get("name") or provider_id or "ESPN"
        ).strip()
        return provider_id, provider_name
    text = str(provider or "ESPN").strip()
    return "", text


def _stage_side(
    item: dict[str, object],
    *,
    team_side: str,
    stage: str,
) -> dict[str, object] | None:
    value = item.get(f"{team_side}TeamOdds")
    if not isinstance(value, dict):
        return None
    stage_value = value.get(stage)
    return stage_value if isinstance(stage_value, dict) else None


def parse_espn_archive_item(
    item: dict[str, object],
    *,
    game_id: str,
    event_id: str,
    stage: str,
) -> list[dict[str, object]]:
    """Parse one provider's explicit archived open or close snapshot."""

    if stage not in {"open", "close"}:
        raise ValueError("stage must be open or close")
    provider_id, provider_name = _provider(item)
    provider_key = canonical_book_identity(provider_name)
    if not provider_key:
        return []
    book_name = "ESPN BET" if provider_key == "espnbet" else provider_name

    output: list[dict[str, object]] = []

    home = _stage_side(item, team_side="home", stage=stage)
    away = _stage_side(item, team_side="away", stage=stage)
    if home is not None and away is not None:
        home_ml = _american(home.get("moneyLine"))
        away_ml = _american(away.get("moneyLine"))
        if home_ml is not None and away_ml is not None:
            for side, odds in (("home", home_ml), ("away", away_ml)):
                output.append(
                    {
                        "game_id": game_id,
                        "market_type": "moneyline",
                        "side": side,
                        "line": None,
                        "american_odds": odds,
                        "provider": "espn_archive",
                        "book": book_name,
                        "canonical_book": provider_key,
                        "provider_id": provider_id,
                        "source_event_id": event_id,
                        "snapshot_id": (
                            f"{event_id}:{provider_id or provider_key}:"
                            f"{stage}:moneyline"
                        ),
                        "archive_stage": stage,
                    }
                )

        home_line = _line(home.get("pointSpread"))
        away_line = _line(away.get("pointSpread"))
        home_price = _american(home.get("spread"))
        away_price = _american(away.get("spread"))
        if (
            home_line is not None
            and away_line is not None
            and home_price is not None
            and away_price is not None
            and abs(home_line + away_line) <= 1e-6
        ):
            for side, line_value, odds in (
                ("home", home_line, home_price),
                ("away", away_line, away_price),
            ):
                output.append(
                    {
                        "game_id": game_id,
                        "market_type": "spread",
                        "side": side,
                        "line": line_value,
                        "american_odds": odds,
                        "provider": "espn_archive",
                        "book": book_name,
                        "canonical_book": provider_key,
                        "provider_id": provider_id,
                        "source_event_id": event_id,
                        "snapshot_id": (
                            f"{event_id}:{provider_id or provider_key}:"
                            f"{stage}:spread"
                        ),
                        "archive_stage": stage,
                    }
                )

    total_stage = item.get(stage)
    if isinstance(total_stage, dict):
        total = _line(total_stage.get("total"))
        over = _american(total_stage.get("over"))
        under = _american(total_stage.get("under"))
        if total is not None and over is not None and under is not None:
            for side, odds in (("over", over), ("under", under)):
                output.append(
                    {
                        "game_id": game_id,
                        "market_type": "total",
                        "side": side,
                        "line": total,
                        "american_odds": odds,
                        "provider": "espn_archive",
                        "book": book_name,
                        "canonical_book": provider_key,
                        "provider_id": provider_id,
                        "source_event_id": event_id,
                        "snapshot_id": (
                            f"{event_id}:{provider_id or provider_key}:"
                            f"{stage}:total"
                        ),
                        "archive_stage": stage,
                    }
                )
    return output


class ESPNArchiveClient:
    """No-key ESPN client for completed-game archived open/close snapshots."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        max_workers: int = 12,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_workers = int(max_workers)
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

    def _json(
        self,
        url: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        target = url if not params else f"{url}?{urlencode(params)}"
        request = Request(
            target,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; HarbinNFLAnalytics/0.1)",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataContractError(
                f"ESPN archived NFL odds request failed: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DataContractError(
                "ESPN archived NFL odds response is not a JSON object"
            )
        return payload

    def scoreboard(self, *, season: int, week: int) -> dict[str, object]:
        return self._json(
            ESPN_SCOREBOARD_URL,
            {
                "limit": 100,
                "dates": str(int(season)),
                "seasontype": 2,
                "week": int(week),
            },
        )

    def core_odds(self, event_id: str) -> list[dict[str, object]]:
        payload = self._json(
            ESPN_CORE_ODDS_URL.format(event_id=event_id),
            {"limit": 50},
        )
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        resolved: list[dict[str, object]] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            item = raw
            reference = raw.get("$ref")
            if reference:
                try:
                    item = self._json(
                        str(reference).replace("http://", "https://")
                    )
                except DataContractError:
                    continue
            resolved.append(item)
        return resolved


def _complete_pairs(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["market_type"]), str(row["canonical_book"]))
        grouped.setdefault(key, []).append(row)
    output: dict[tuple[str, str], list[dict[str, object]]] = {}
    for key, pair in grouped.items():
        if len(pair) != 2:
            continue
        sides = {str(row["side"]) for row in pair}
        market = key[0]
        expected = (
            {"home", "away"}
            if market in {"moneyline", "spread"}
            else {"over", "under"}
        )
        if sides != expected:
            continue
        output[key] = pair
    return output


def _provider_rank(pair: list[dict[str, object]]) -> tuple[int, str]:
    name = str(pair[0]["book"])
    return (1 if "live odds" in name.lower() else 0, name.lower())


def fetch_espn_archive_quotes(
    projections: pl.DataFrame,
    *,
    client: ESPNArchiveClient | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, ESPNArchiveCoverage]:
    """Fetch archived open/close quotes for the projection games."""

    require_columns(projections, PROJECTION_REQUIRED, "espn_archive_projections")
    if projections.is_empty():
        raise DataContractError("espn_archive_projections is empty")

    source = client or ESPNArchiveClient()
    entry_rows: list[dict[str, object]] = []
    closing_rows: list[dict[str, object]] = []
    errors: list[str] = []
    matched_events = 0
    games_with_open: set[str] = set()
    games_with_close: set[str] = set()

    groups = projections.select("season", "week").unique().sort(["season", "week"])
    for group in groups.iter_rows(named=True):
        season = int(group["season"])
        week = int(group["week"])
        targets = projections.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        target_map = {
            (str(row["home_team"]), str(row["away_team"])): str(row["game_id"])
            for row in targets.iter_rows(named=True)
        }
        try:
            scoreboard = source.scoreboard(season=season, week=week)
        except DataContractError as exc:
            errors.append(f"{season}W{week} scoreboard: {exc}")
            continue
        events = scoreboard.get("events")
        if not isinstance(events, list):
            errors.append(f"{season}W{week} scoreboard missing events")
            continue

        matched: list[tuple[str, str]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            teams = _event_teams(event)
            if teams is None:
                continue
            game_id = target_map.get(teams)
            event_id = str(event.get("id") or "").strip()
            if game_id is None or not event_id:
                continue
            matched.append((game_id, event_id))
        matched_events += len(matched)

        with ThreadPoolExecutor(max_workers=source.max_workers) as executor:
            futures = {
                executor.submit(source.core_odds, event_id): (game_id, event_id)
                for game_id, event_id in matched
            }
            for future in as_completed(futures):
                game_id, event_id = futures[future]
                try:
                    items = future.result()
                except DataContractError as exc:
                    errors.append(f"{game_id} core: {exc}")
                    continue

                per_stage: dict[
                    str,
                    dict[
                        tuple[str, str],
                        tuple[tuple[int, str], list[dict[str, object]]],
                    ],
                ] = {"open": {}, "close": {}}
                for item in items:
                    for stage in ("open", "close"):
                        parsed = parse_espn_archive_item(
                            item,
                            game_id=game_id,
                            event_id=event_id,
                            stage=stage,
                        )
                        for key, pair in _complete_pairs(parsed).items():
                            candidate = (_provider_rank(pair), pair)
                            previous = per_stage[stage].get(key)
                            if previous is None or candidate[0] < previous[0]:
                                per_stage[stage][key] = candidate

                opens = [
                    row
                    for _, pair in per_stage["open"].values()
                    for row in pair
                ]
                closes = [
                    row
                    for _, pair in per_stage["close"].values()
                    for row in pair
                ]
                if opens:
                    games_with_open.add(game_id)
                    entry_rows.extend(opens)
                if closes:
                    games_with_close.add(game_id)
                    closing_rows.extend(closes)

    entry = (
        pl.DataFrame(entry_rows).sort(
            ["game_id", "book", "market_type", "side"]
        )
        if entry_rows
        else pl.DataFrame()
    )
    closing = (
        pl.DataFrame(closing_rows).sort(
            ["game_id", "book", "market_type", "side"]
        )
        if closing_rows
        else pl.DataFrame()
    )
    books = sorted(
        {
            str(value)
            for frame in (entry, closing)
            if not frame.is_empty()
            for value in frame.get_column("book").unique().to_list()
        }
    )
    coverage = ESPNArchiveCoverage(
        requested_games=projections.get_column("game_id").n_unique(),
        matched_events=matched_events,
        games_with_open=len(games_with_open),
        games_with_close=len(games_with_close),
        entry_pairs=entry.height // 2 if not entry.is_empty() else 0,
        closing_pairs=closing.height // 2 if not closing.is_empty() else 0,
        books=tuple(books),
        source_errors=tuple(errors[-50:]),
    )
    return entry, closing, coverage


def _compare_archive_entries(
    projections: pl.DataFrame,
    entries: pl.DataFrame,
    *,
    min_probability_training_games: int,
) -> pl.DataFrame:
    if entries.is_empty():
        return pl.DataFrame()
    require_columns(entries, ARCHIVE_QUOTE_REQUIRED, "espn_archive_entries")

    output: list[dict[str, object]] = []
    entry_game_ids = entries.get_column("game_id").unique().to_list()
    target_projections = projections.filter(
        pl.col("game_id").is_in(entry_game_ids)
    )
    groups = (
        target_projections.select("season", "week")
        .unique()
        .sort(["season", "week"])
    )
    for group in groups.iter_rows(named=True):
        season = int(group["season"])
        week = int(group["week"])
        history = projections.filter(
            (pl.col("season") < season)
            | ((pl.col("season") == season) & (pl.col("week") < week))
        )
        if history.height < min_probability_training_games:
            continue
        targets = projections.filter(
            (pl.col("season") == season) & (pl.col("week") == week)
        )
        projection_map = {
            str(row["game_id"]): row
            for row in targets.iter_rows(named=True)
        }
        week_entries = entries.filter(
            pl.col("game_id").is_in(list(projection_map))
        )
        grouped: dict[
            tuple[str, str, str, str],
            list[dict[str, object]],
        ] = {}
        for row in week_entries.iter_rows(named=True):
            key = (
                str(row["game_id"]),
                str(row["book"]),
                str(row["market_type"]),
                str(row["snapshot_id"]),
            )
            grouped.setdefault(key, []).append(row)

        distribution, probability_meta = (
            build_operational_probability_distribution(
                history,
                current_season=season,
            )
        )
        for rows in grouped.values():
            if len(rows) != 2:
                continue
            game_id = str(rows[0]["game_id"])
            projection = projection_map.get(game_id)
            if projection is None:
                continue
            quotes = [
                MarketQuote(
                    market_type=row["market_type"],
                    side=row["side"],
                    line=(
                        None
                        if row["line"] is None
                        else float(row["line"])
                    ),
                    american_odds=int(row["american_odds"]),
                    book=str(row["book"]),
                )
                for row in rows
            ]
            try:
                comparisons = compare_two_way_market(
                    distribution,
                    projected_home_margin=float(
                        projection["projected_home_margin"]
                    ),
                    projected_total=float(projection["projected_total"]),
                    first=quotes[0],
                    second=quotes[1],
                )
            except ValueError:
                continue
            source_by_side = {
                str(row["side"]): row
                for row in rows
            }
            for comparison in comparisons:
                source_row = source_by_side[comparison.side]
                result = comparison.to_dict()
                result.update(
                    {
                        "game_id": game_id,
                        "provider": source_row["provider"],
                        "book": source_row["book"],
                        "captured_at": None,
                        "snapshot_id": source_row["snapshot_id"],
                        "source_event_id": source_row["source_event_id"],
                        "decision_time": None,
                        "projected_home_margin": float(
                            projection["projected_home_margin"]
                        ),
                        "projected_total": float(
                            projection["projected_total"]
                        ),
                        "probability_model_family": probability_meta.get(
                            "model_family"
                        ),
                        "probability_reliability_ready": bool(
                            probability_meta.get("reliability_ready", False)
                        ),
                    }
                )
                output.append(result)
    return (
        pl.DataFrame(output).sort(
            ["game_id", "book", "market_type", "side"]
        )
        if output
        else pl.DataFrame()
    )


def _choose_best(comparisons: pl.DataFrame) -> pl.DataFrame:
    if comparisons.is_empty():
        return comparisons
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in comparisons.iter_rows(named=True):
        grouped.setdefault(
            (str(row["game_id"]), str(row["market_type"])),
            [],
        ).append(row)

    rows: list[dict[str, object]] = []
    for candidates in grouped.values():
        books = {str(row["book"]) for row in candidates}
        chosen = max(
            candidates,
            key=lambda row: (
                float(row["expected_value_per_unit"]),
                float(row["probability_edge"]),
                int(row["american_odds"]),
                str(row["book"]),
                str(row["side"]),
            ),
        )
        result = dict(chosen)
        result["market_book_count"] = len(books)
        rows.append(result)
    return pl.DataFrame(rows).sort(["game_id", "market_type"])


def _pair_map(
    frame: pl.DataFrame,
) -> dict[tuple[str, str, str], dict[str, dict[str, object]]]:
    output: dict[
        tuple[str, str, str],
        dict[str, dict[str, object]],
    ] = {}
    if frame.is_empty():
        return output
    for row in frame.iter_rows(named=True):
        key = (
            str(row["game_id"]),
            str(row["book"]),
            str(row["market_type"]),
        )
        output.setdefault(key, {})[str(row["side"])] = row
    return output


def _clv(
    entry: dict[str, object],
    entry_pair: dict[str, dict[str, object]],
    close_pair: dict[str, dict[str, object]],
) -> float | None:
    market = str(entry["market_type"])
    side = str(entry["side"])
    close = close_pair.get(side)
    if close is None:
        return None
    if market == "spread":
        if entry.get("line") is None or close.get("line") is None:
            return None
        return float(entry["line"]) - float(close["line"])
    if market == "total":
        if entry.get("line") is None or close.get("line") is None:
            return None
        if side == "over":
            return float(close["line"]) - float(entry["line"])
        return float(entry["line"]) - float(close["line"])
    if market != "moneyline":
        return None
    if set(entry_pair) != {"home", "away"}:
        return None
    if set(close_pair) != {"home", "away"}:
        return None
    entry_home, entry_away = remove_two_way_vig(
        int(entry_pair["home"]["american_odds"]),
        int(entry_pair["away"]["american_odds"]),
    )
    close_home, close_away = remove_two_way_vig(
        int(close_pair["home"]["american_odds"]),
        int(close_pair["away"]["american_odds"]),
    )
    return (
        close_home - entry_home
        if side == "home"
        else close_away - entry_away
    )


def build_espn_archive_bets(
    projections: pl.DataFrame,
    entries: pl.DataFrame,
    closings: pl.DataFrame,
    *,
    min_probability_training_games: int = 64,
) -> pl.DataFrame:
    """Grade one best opening side per game/market and attach archived close CLV."""

    require_columns(projections, PROJECTION_REQUIRED, "espn_archive_projections")
    if min_probability_training_games < 64:
        raise ValueError("min_probability_training_games must be >= 64")
    comparisons = _compare_archive_entries(
        projections,
        entries,
        min_probability_training_games=min_probability_training_games,
    )
    chosen = _choose_best(comparisons)
    if chosen.is_empty():
        return chosen

    graded = grade_market_comparisons(
        chosen,
        projections.select(
            "game_id",
            "actual_home_margin",
            "actual_total",
        ),
    )
    metadata = projections.select(
        "season",
        "week",
        "game_id",
        "home_team",
        "away_team",
    ).unique(subset=["game_id"])
    graded = graded.join(metadata, on="game_id", how="left")

    entry_pairs = _pair_map(entries)
    close_pairs = _pair_map(closings)
    rows: list[dict[str, object]] = []
    for source in graded.iter_rows(named=True):
        row = dict(source)
        key = (
            str(row["game_id"]),
            str(row["book"]),
            str(row["market_type"]),
        )
        close_pair = close_pairs.get(key, {})
        close = close_pair.get(str(row["side"]))
        row["clv_proxy"] = _clv(
            row,
            entry_pairs.get(key, {}),
            close_pair,
        )
        row["entry_line_observed"] = True
        row["entry_price_verified"] = True
        row["entry_quote_verified"] = True
        row["entry_timestamp_verified"] = False
        row["entry_price_stage"] = "espn_archived_open"
        row["price_stage"] = "espn_archived_open"
        row["historical_provenance"] = "provider_labeled_open_close"
        row["closing_quote_verified"] = close is not None
        row["closing_price_stage"] = "espn_archived_close"
        row["closing_snapshot_at"] = None
        row["closing_line"] = None if close is None else close.get("line")
        row["closing_odds"] = (
            None if close is None else close.get("american_odds")
        )
        rows.append(row)
    return pl.DataFrame(rows).sort(
        ["season", "week", "game_id", "market_type"]
    )


def build_espn_archive_report(
    bets: pl.DataFrame,
    coverage: ESPNArchiveCoverage,
) -> dict[str, object]:
    overall = (
        summarize_archive_bets(bets).to_dict()
        if not bets.is_empty()
        else {}
    )
    clv_samples = int(overall.get("clv_samples", 0) or 0)
    clv_coverage = clv_samples / bets.height if bets.height else 0.0

    by_market: dict[str, object] = {}
    by_season: dict[str, object] = {}
    if not bets.is_empty():
        for market in bets.get_column("market_type").unique().sort().to_list():
            by_market[str(market)] = summarize_archive_bets(
                bets.filter(pl.col("market_type") == market)
            ).to_dict()
        for season in bets.get_column("season").unique().sort().to_list():
            by_season[str(season)] = summarize_archive_bets(
                bets.filter(pl.col("season") == season)
            ).to_dict()

    return {
        "version": 1,
        "status": "READY" if bets.height else "EMPTY",
        "provider": "espn_archive",
        "timestamped_entry_prices": False,
        "archived_open_close_verified": True,
        "entry_provenance": "provider_labeled_open",
        "closing_provenance": "provider_labeled_close",
        "bets": bets.height,
        "clv_samples": clv_samples,
        "clv_coverage": clv_coverage,
        "coverage": coverage.to_dict(),
        "overall": overall,
        "by_market": by_market,
        "by_season": by_season,
        "meaning": (
            "Historical promotion sample uses ESPN's explicit archived provider open "
            "and close snapshots. These are verified archive stages, not fabricated "
            "point-in-time timestamps. Forward evidence retains strict timestamped "
            "capture requirements."
        ),
    }
