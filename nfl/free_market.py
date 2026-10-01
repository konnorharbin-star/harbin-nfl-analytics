"""Free NFL market archive built from nflverse public data.

This mirrors the CFB platform's free historical-market philosophy. The canonical
football model remains independent of sportsbook prices. Archived market fields are
used only after a fair score/probability already exists.

nflverse schedules contain archived final spread, total, moneyline and price fields.
The separate ``initial_lines.csv`` file can supply opening spread/total observations
when available. Because the free initial-lines file does not carry timestamps or every
season/market, any opening-to-final movement is explicitly labeled a CLV *proxy*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import polars as pl

from .contracts import DataContractError, require_columns, require_unique

NFLVERSE_INITIAL_LINES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/initial_lines.csv"
)

SCHEDULE_MARKET_REQUIRED = {
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
}
INITIAL_LINE_REQUIRED = {"season", "sportsbook", "type", "about", "side", "line"}


@dataclass(frozen=True)
class FreeArchiveQuote:
    game_id: str
    season: int
    week: int
    provider: str
    archive_book: str
    opening_book: str | None
    open_home_ml: int | None
    open_away_ml: int | None
    final_home_ml: int | None
    final_away_ml: int | None
    open_home_spread: float | None
    final_home_spread: float | None
    open_home_spread_odds: int | None
    open_away_spread_odds: int | None
    final_home_spread_odds: int | None
    final_away_spread_odds: int | None
    open_total: float | None
    final_total: float | None
    open_over_odds: int | None
    open_under_odds: int | None
    final_over_odds: int | None
    final_under_odds: int | None
    has_distinct_open_spread: bool
    has_distinct_open_total: bool
    has_distinct_open_moneyline: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def has_any_distinct_open(self) -> bool:
        return (
            self.has_distinct_open_spread
            or self.has_distinct_open_total
            or self.has_distinct_open_moneyline
        )


def _american(value: object, *, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if numeric != numeric:
        return fallback
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        raise DataContractError(f"archive American odds are non-integral: {numeric}")
    if rounded == 0 or -100 < rounded < 100:
        return fallback
    return rounded


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if numeric != numeric else numeric


def validate_initial_lines(frame: pl.DataFrame) -> None:
    require_columns(frame, INITIAL_LINE_REQUIRED, "nflverse_initial_lines")
    if frame.is_empty():
        raise DataContractError("nflverse_initial_lines is empty")
    allowed = frame.get_column("type").cast(pl.String).str.to_uppercase()
    if frame.filter(~allowed.is_in(["SPREAD", "TOTAL", "MONEYLINE"])).height:
        raise DataContractError("nflverse_initial_lines contains an unsupported market type")


def load_nflverse_initial_lines(
    *,
    cache_path: str | Path = "data/cache/nflverse_initial_lines.csv",
    refresh: bool = False,
    timeout_seconds: float = 30.0,
) -> pl.DataFrame:
    """Download/cache nflverse's free opening-line file.

    The file is public and requires no API key. Tests should inject a fixture instead of
    performing network access.
    """

    path = Path(cache_path)
    if path.exists() and not refresh:
        frame = pl.read_csv(path)
        validate_initial_lines(frame)
        return frame

    request = Request(
        NFLVERSE_INITIAL_LINES_URL,
        headers={"User-Agent": "harbin-nfl-analytics/0.1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DataContractError(f"failed to download nflverse initial lines: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    frame = pl.read_csv(path)
    validate_initial_lines(frame)
    return frame


class FreeNFLMarketStore:
    """NCAA-style free archive market store for NFL backtests."""

    def __init__(
        self,
        schedules: pl.DataFrame,
        *,
        initial_lines: pl.DataFrame | None = None,
    ) -> None:
        require_columns(schedules, SCHEDULE_MARKET_REQUIRED, "nflverse_schedules")
        require_unique(schedules, ["game_id"], "nflverse_schedules")
        self.schedules = schedules
        self.initial_lines = initial_lines
        if initial_lines is not None:
            validate_initial_lines(initial_lines)

    def _schedule_row(self, game_id: str) -> dict[str, object] | None:
        rows = self.schedules.filter(pl.col("game_id").cast(pl.String) == str(game_id))
        if rows.is_empty():
            return None
        return rows.row(0, named=True)

    def _opening_values(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
    ) -> dict[str, object]:
        empty: dict[str, object] = {
            "book": None,
            "home_spread": None,
            "total": None,
            "home_ml": None,
            "away_ml": None,
        }
        if self.initial_lines is None:
            return empty

        rows = self.initial_lines.filter(pl.col("about").cast(pl.String) == str(game_id))
        if rows.is_empty():
            return empty

        candidates: list[tuple[int, str]] = []
        for book_rows in rows.partition_by("sportsbook", maintain_order=True):
            book = str(book_rows.get_column("sportsbook")[0])
            market_count = (
                book_rows.get_column("type")
                .cast(pl.String)
                .str.to_uppercase()
                .n_unique()
            )
            candidates.append((-int(market_count), book))
        candidates.sort()
        chosen = candidates[0][1]
        rows = rows.filter(pl.col("sportsbook").cast(pl.String) == chosen).with_columns(
            pl.col("type").cast(pl.String).str.to_uppercase().alias("_type"),
            pl.col("side").cast(pl.String).alias("_side"),
        )

        spread = rows.filter(pl.col("_type") == "SPREAD")
        home_spread: float | None = None
        if not spread.is_empty():
            home = spread.filter(pl.col("_side") == home_team)
            away = spread.filter(pl.col("_side") == away_team)
            if not home.is_empty():
                home_spread = _number(home.tail(1).get_column("line")[0])
            elif not away.is_empty():
                away_line = _number(away.tail(1).get_column("line")[0])
                home_spread = None if away_line is None else -away_line

        total_rows = rows.filter(pl.col("_type") == "TOTAL")
        total: float | None = None
        if not total_rows.is_empty():
            over = total_rows.filter(pl.col("_side").str.to_lowercase() == "over")
            source = over.tail(1) if not over.is_empty() else total_rows.tail(1)
            total = _number(source.get_column("line")[0])

        moneyline = rows.filter(pl.col("_type") == "MONEYLINE")
        home_ml: int | None = None
        away_ml: int | None = None
        if not moneyline.is_empty():
            home = moneyline.filter(pl.col("_side") == home_team)
            away = moneyline.filter(pl.col("_side") == away_team)
            if not home.is_empty():
                home_ml = _american(home.tail(1).get_column("line")[0])
            if not away.is_empty():
                away_ml = _american(away.tail(1).get_column("line")[0])

        return {
            "book": chosen,
            "home_spread": home_spread,
            "total": total,
            "home_ml": home_ml,
            "away_ml": away_ml,
        }

    def quote(self, game_id: str) -> FreeArchiveQuote | None:
        """Return free opening/archive-final market values for one NFL game."""

        row = self._schedule_row(game_id)
        if row is None:
            return None
        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # nflverse ``spread_line`` is positive when the home team is favored and
        # negative when the away team is favored. A bettable home handicap therefore
        # has the opposite sign.
        spread_line = _number(row["spread_line"])
        final_home_spread = None if spread_line is None else -spread_line
        final_total = _number(row["total_line"])
        final_home_ml = _american(row["home_moneyline"])
        final_away_ml = _american(row["away_moneyline"])
        final_home_spread_odds = _american(row["home_spread_odds"], fallback=-110)
        final_away_spread_odds = _american(row["away_spread_odds"], fallback=-110)
        final_over_odds = _american(row["over_odds"], fallback=-110)
        final_under_odds = _american(row["under_odds"], fallback=-110)

        opening = self._opening_values(str(game_id), home_team, away_team)
        open_home_spread = _number(opening["home_spread"])
        open_total = _number(opening["total"])
        open_home_ml = _american(opening["home_ml"])
        open_away_ml = _american(opening["away_ml"])

        distinct_spread = open_home_spread is not None and final_home_spread is not None
        distinct_total = open_total is not None and final_total is not None
        distinct_ml = open_home_ml is not None and open_away_ml is not None

        if open_home_spread is None:
            open_home_spread = final_home_spread
        if open_total is None:
            open_total = final_total
        if open_home_ml is None:
            open_home_ml = final_home_ml
        if open_away_ml is None:
            open_away_ml = final_away_ml

        if not any(
            value is not None
            for value in (
                open_home_ml,
                open_away_ml,
                open_home_spread,
                open_total,
            )
        ):
            return None

        return FreeArchiveQuote(
            game_id=str(game_id),
            season=int(row["season"]),
            week=int(row["week"]),
            provider="nflverse",
            archive_book="nflverse_archive",
            opening_book=None if opening["book"] is None else str(opening["book"]),
            open_home_ml=open_home_ml,
            open_away_ml=open_away_ml,
            final_home_ml=final_home_ml,
            final_away_ml=final_away_ml,
            open_home_spread=open_home_spread,
            final_home_spread=final_home_spread,
            open_home_spread_odds=final_home_spread_odds,
            open_away_spread_odds=final_away_spread_odds,
            final_home_spread_odds=final_home_spread_odds,
            final_away_spread_odds=final_away_spread_odds,
            open_total=open_total,
            final_total=final_total,
            open_over_odds=final_over_odds,
            open_under_odds=final_under_odds,
            final_over_odds=final_over_odds,
            final_under_odds=final_under_odds,
            has_distinct_open_spread=distinct_spread,
            has_distinct_open_total=distinct_total,
            has_distinct_open_moneyline=distinct_ml,
        )
