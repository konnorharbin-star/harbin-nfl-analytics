"""CFB-style human-readable NFL weekly picks board."""

from __future__ import annotations

import html
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from PIL import Image, ImageDraw, ImageFont

BG = (15, 17, 19)
ALT = (22, 24, 27)
GRID = (39, 42, 46)
TEXT = (240, 241, 242)
MUTED = (128, 132, 136)
BLUE = (76, 132, 224)
GREEN = (95, 196, 104)
DARK_GREEN = (31, 70, 42)
AMBER = (72, 59, 30)
AMBER_TEXT = (227, 181, 73)
WARN = (214, 164, 75)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _number(value: object) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_odds(value: object) -> str:
    number = _number(value)
    if number is None:
        return "—"
    integer = int(round(number))
    return f"+{integer}" if integer > 0 else str(integer)


def _fmt_line(value: object) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:+g}"


def _display_updated_at(value: str) -> str:
    """Render canonical UTC timestamps like the CFB board's Central-time header."""

    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if stamp.tzinfo is None:
        return value
    local = stamp.astimezone(ZoneInfo("America/Chicago"))
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%b')} {local.day}, {local.year} · {clock} CT"


def _team_side(row: dict[str, object]) -> str:
    side = str(row.get("quant_side") or "")
    if side == "home":
        return str(row.get("home_team") or "HOME")
    if side == "away":
        return str(row.get("away_team") or "AWAY")
    if side == "over":
        return "O"
    if side == "under":
        return "U"
    return side.upper()


def _signal(row: dict[str, object] | None) -> str:
    if not row:
        return ""
    value = str(row.get("quant_signal") or "PASS").upper()
    return "" if value == "PASS" else value


def _moneyline_text(row: dict[str, object] | None) -> str | None:
    if not row or _number(row.get("quant_odds")) is None:
        return None
    return f"{_team_side(row)} {_fmt_odds(row.get('quant_odds'))}"


def _spread_text(row: dict[str, object] | None) -> str | None:
    if not row or _number(row.get("quant_price")) is None:
        return None
    return f"{_team_side(row)} {_fmt_line(row.get('quant_price'))}"


def _total_text(
    row: dict[str, object] | None,
    *,
    projected_total: float,
) -> tuple[str | None, str]:
    projection = f"proj {int(round(projected_total))}"
    if not row or _number(row.get("quant_price")) is None:
        return None, projection
    line = _number(row.get("quant_price"))
    assert line is not None
    return f"{_team_side(row)} {line:g}", projection


def build_weekly_board(current: pl.DataFrame) -> pl.DataFrame:
    """Pivot one-row-per-market canonical output into one publication row per game."""

    if current.is_empty():
        return pl.DataFrame()
    required = {
        "game_id",
        "week",
        "away_team",
        "home_team",
        "model_margin_home",
        "model_total",
        "quant_market",
    }
    missing = sorted(required.difference(current.columns))
    if missing:
        raise ValueError(f"weekly renderer missing column(s): {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    game_ids = current.get_column("game_id").unique().sort().to_list()
    for game_id in game_ids:
        group = current.filter(pl.col("game_id") == game_id)
        values = group.to_dicts()
        base = values[0]
        margin = float(base["model_margin_home"])
        total = float(base["model_total"])
        home_points = (total + margin) / 2.0
        away_points = (total - margin) / 2.0
        home_probability = _number(base.get("calibrated_home_probability"))
        if home_probability is None:
            home_probability = 0.5
        winner = str(base["home_team"] if margin >= 0 else base["away_team"])
        winner_probability = home_probability if margin >= 0 else 1.0 - home_probability
        markets = {str(value.get("quant_market")): value for value in values}
        rows.append(
            {
                "game_id": str(game_id),
                "week": int(base["week"]),
                "date": base.get("date"),
                "kickoff": base.get("kickoff"),
                "away_team": str(base["away_team"]),
                "home_team": str(base["home_team"]),
                "away_score": int(round(away_points)),
                "home_score": int(round(home_points)),
                "winner": winner,
                "win_pct": int(round(100.0 * winner_probability)),
                "proj_total": total,
                "moneyline": markets.get("moneyline"),
                "spread": markets.get("spread"),
                "total": markets.get("total"),
            }
        )
    return pl.DataFrame(rows).sort(["date", "kickoff", "game_id"])


def _badge_html(label: str) -> str:
    if not label:
        return ""
    return f'<span class="badge {label.lower()}">{html.escape(label)}</span>'


def _market_html(
    row: dict[str, object] | None,
    *,
    market: str,
    projected_total: float,
) -> str:
    if market == "moneyline":
        text = _moneyline_text(row)
        projection = ""
    elif market == "spread":
        text = _spread_text(row)
        projection = ""
    elif market == "total":
        text, projection = _total_text(row, projected_total=projected_total)
    else:
        raise ValueError(f"unknown publication market: {market}")

    signal = _signal(row)
    if text is None:
        quiet = '<span class="quiet">NO LINE</span>'
        if projection:
            quiet += f' <span class="projtot">{html.escape(projection)}</span>'
        return quiet

    parts = [html.escape(text)]
    if projection:
        parts.append(f'<span class="projtot">{html.escape(projection)}</span>')
    if signal:
        parts.append(_badge_html(signal))
    return " ".join(parts)


def _has_live_market(board: pl.DataFrame) -> bool:
    if board.is_empty():
        return False
    for row in board.to_dicts():
        if any(row.get(name) for name in ("moneyline", "spread", "total")):
            return True
    return False


def render_html(
    board: pl.DataFrame,
    path: str | Path,
    *,
    week: int,
    updated_at: str,
    release_state: str,
) -> None:
    state = release_state.upper()
    status = "Live market data" if _has_live_market(board) else "Projection-only · no verified live lines"
    if state != "PRODUCTION":
        status = f"{status} · {state} validation"

    css = """*{box-sizing:border-box}body{margin:0;background:#0f1113;color:#f0f1f2;font-family:Inter,Arial,sans-serif}.shell{max-width:1320px;margin:auto;padding:14px}.page{display:none}.head{display:flex;justify-content:space-between;align-items:flex-start}.title{font-size:24px;font-weight:800}.sub,.counter{font-size:12px;color:#8b8e92}.status{font-size:10px;color:#aeb1b5;margin-top:3px}.status.warn{color:#d6a44b}.counter{text-align:right;line-height:1.45}table{width:100%;border-collapse:collapse;table-layout:fixed;margin-top:8px}th{font-size:9px;letter-spacing:.08em;color:#777c81;text-align:left;padding:7px 8px;border-bottom:1px solid #272a2e}td{font-size:12px;padding:8px;border-bottom:1px solid #272a2e;height:36px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}tbody tr:nth-child(even){background:#16181b}.winner{font-weight:800}.at{color:#777c81}.proj{font-weight:800}.bar{display:inline-block;height:6px;background:#4c84e0;border-radius:5px;margin-right:6px;vertical-align:middle}.quiet{color:#64686d}.active{color:#f0f1f2;font-weight:700}.projtot{font-size:10px;color:#96999d}.badge{font-size:8px;font-weight:800;border-radius:4px;padding:3px 5px;margin-left:4px}.strong{background:#5fc468;color:#0c2c12}.bet{background:#1f462a;color:#63c76d}.lean{background:#483b1e;color:#e3b549}.nav{text-align:center;padding:14px}.nav button{background:#202328;border:1px solid #373b40;color:#eee;border-radius:5px;padding:6px 10px;margin:2px}.c1{width:23%}.c2{width:7%}.c3{width:9%}.c4{width:23%}.c5{width:23%}.c6{width:15%}@media(max-width:900px){.shell{overflow-x:auto}.page{min-width:1100px}}"""

    values = board.to_dicts()
    pages: list[str] = []
    page_count = max(1, math.ceil(len(values) / 14))
    warn_class = "warn" if not _has_live_market(board) else ""
    formatted_updated_at = _display_updated_at(updated_at)

    for page in range(1, page_count + 1):
        start = (page - 1) * 14
        chunk = values[start : start + 14]
        rows: list[str] = []
        for row in chunk:
            away = html.escape(str(row["away_team"]))
            home = html.escape(str(row["home_team"]))
            winner = str(row["winner"])
            matchup = (
                f'<span class="winner">{away}</span>'
                if row["away_team"] == winner
                else away
            )
            matchup += ' <span class="at">@</span> '
            matchup += (
                f'<span class="winner">{home}</span>'
                if row["home_team"] == winner
                else home
            )
            pct = int(row["win_pct"])
            bar = max(4, min(52, int((pct - 50) * 1.15)))
            rows.append(
                "<tr>"
                f"<td>{matchup}</td>"
                f"<td class=\"proj\">{row['away_score']}–{row['home_score']}</td>"
                f"<td><span class=\"bar\" style=\"width:{bar}px\"></span>{pct}%</td>"
                f"<td>{_market_html(row.get('moneyline'), market='moneyline', projected_total=float(row['proj_total']))}</td>"
                f"<td>{_market_html(row.get('spread'), market='spread', projected_total=float(row['proj_total']))}</td>"
                f"<td>{_market_html(row.get('total'), market='total', projected_total=float(row['proj_total']))}</td>"
                "</tr>"
            )
        game_range = (
            f"Games {start + 1}–{min(start + 14, len(values))} of {len(values)}"
            if values
            else "0 games"
        )
        pages.append(
            f'<section class="page" id="p{page}"><div class="head"><div>'
            f'<div class="title">NFL MODEL · WEEK {week} PICKS</div>'
            f'<div class="sub">Projected scores &amp; best bets · Updated {html.escape(formatted_updated_at)}</div>'
            f'<div class="status {warn_class}">{html.escape(status)}</div></div>'
            f'<div class="counter">{game_range}<br>{page} / {page_count}</div></div>'
            '<table><colgroup><col class="c1"><col class="c2"><col class="c3">'
            '<col class="c4"><col class="c5"><col class="c6"></colgroup>'
            '<thead><tr><th>MATCHUP (WINNER BOLD)</th><th>PROJ</th><th>WIN %</th>'
            '<th>MONEYLINE</th><th>SPREAD</th><th>TOTAL</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )

    nav = "".join(
        f'<button onclick="show({page})">{page}</button>'
        for page in range(1, page_count + 1)
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>NFL Model Week {week}</title><style>{css}</style></head><body>'
        f'<div class="shell">{"".join(pages)}<div class="nav">{nav}</div></div>'
        '<script>function show(n){document.querySelectorAll(".page").forEach('
        '(e,i)=>e.style.display=i===n-1?"block":"none")}show(1)</script></body></html>'
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")


def _text_width(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
) -> int:
    return draw.textbbox((0, 0), value, font=font)[2]


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
) -> int:
    if not label:
        return 0
    width = _text_width(draw, label, font) + 12
    fill = GREEN if label == "STRONG" else DARK_GREEN if label == "BET" else AMBER
    text = (12, 45, 17) if label == "STRONG" else GREEN if label == "BET" else AMBER_TEXT
    draw.rounded_rectangle((x, y - 2, x + width, y + 13), radius=4, fill=fill)
    draw.text((x + 6, y + 1), label, font=font, fill=text)
    return width + 5


def _draw_market(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    row: dict[str, object] | None,
    market: str,
    projected_total: float,
    row_font: ImageFont.ImageFont,
    bold_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    badge_font: ImageFont.ImageFont,
) -> None:
    if market == "moneyline":
        text = _moneyline_text(row)
        projection = ""
    elif market == "spread":
        text = _spread_text(row)
        projection = ""
    elif market == "total":
        text, projection = _total_text(row, projected_total=projected_total)
    else:
        raise ValueError(f"unknown publication market: {market}")

    signal = _signal(row)
    if text is None:
        draw.text((x, y), "NO LINE", font=small_font, fill=(96, 100, 105))
        if projection:
            offset = _text_width(draw, "NO LINE", small_font) + 8
            draw.text((x + offset, y), projection, font=small_font, fill=(143, 147, 151))
        return

    active = bool(signal)
    font = bold_font if active else row_font
    draw.text((x, y), text, font=font, fill=TEXT if active else (96, 100, 105))
    next_x = x + _text_width(draw, text, font) + 6
    if projection:
        draw.text((next_x, y + 2), projection, font=small_font, fill=(143, 147, 151))
        next_x += _text_width(draw, projection, small_font) + 5
    if active:
        _draw_badge(draw, next_x, y, signal, badge_font)


def render_png(
    board: pl.DataFrame,
    path: str | Path,
    *,
    page: int,
    week: int,
    updated_at: str,
    release_state: str,
) -> None:
    width, height = 1320, 690
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(22, bold=True)
    sub_font = _font(11)
    status_font = _font(9, bold=True)
    head_font = _font(9, bold=True)
    row_font = _font(12)
    bold_font = _font(12, bold=True)
    small_font = _font(9)
    badge_font = _font(8, bold=True)

    values = board.to_dicts()
    page_count = max(1, math.ceil(len(values) / 14))
    page = max(1, min(page, page_count))
    start = (page - 1) * 14
    end = min(start + 14, len(values))
    state = release_state.upper()
    live_market = _has_live_market(board)
    status = "Live market data" if live_market else "Projection-only · no verified live lines"
    if state != "PRODUCTION":
        status = f"{status} · {state} validation"

    draw.text((17, 13), f"NFL MODEL · WEEK {week} PICKS", font=title_font, fill=TEXT)
    draw.text(
        (17, 43),
        f"Projected scores & best bets · Updated {_display_updated_at(updated_at)}",
        font=sub_font,
        fill=MUTED,
    )
    draw.text(
        (17, 59),
        status,
        font=status_font,
        fill=MUTED if live_market else WARN,
    )
    draw.text(
        (1138, 17),
        f"Games {start + 1 if values else 0}–{end} of {len(values)}",
        font=sub_font,
        fill=MUTED,
    )
    draw.text((1260, 39), f"{page} / {page_count}", font=sub_font, fill=MUTED)

    columns = [17, 305, 385, 500, 790, 1075]
    headings = ["MATCHUP (WINNER BOLD)", "PROJ", "WIN %", "MONEYLINE", "SPREAD", "TOTAL"]
    for x, heading in zip(columns, headings, strict=True):
        draw.text((x, 78), heading, font=head_font, fill=(119, 124, 129))
    draw.line((16, 96, 1304, 96), fill=GRID)

    for index, row in enumerate(values[start:end]):
        y = 97 + index * 41
        if index % 2:
            draw.rectangle((16, y, 1304, y + 40), fill=ALT)
        draw.line((16, y, 1304, y), fill=GRID)
        text_y = y + 13
        away = str(row["away_team"])
        home = str(row["home_team"])
        winner = str(row["winner"])
        away_font = bold_font if away == winner else row_font
        home_font = bold_font if home == winner else row_font

        draw.text((17, text_y), away, font=away_font, fill=TEXT)
        next_x = 17 + _text_width(draw, away, away_font)
        draw.text((next_x, text_y), " @ ", font=row_font, fill=MUTED)
        next_x += _text_width(draw, " @ ", row_font)
        draw.text((next_x, text_y), home, font=home_font, fill=TEXT)

        draw.text(
            (305, text_y),
            f"{row['away_score']}–{row['home_score']}",
            font=bold_font,
            fill=TEXT,
        )

        pct = int(row["win_pct"])
        bar_width = max(4, min(46, int((pct - 50) * 1.05)))
        draw.rounded_rectangle(
            (385, text_y + 6, 385 + bar_width, text_y + 11),
            radius=3,
            fill=BLUE,
        )
        draw.text(
            (385 + bar_width + 7, text_y - 1),
            f"{pct}%",
            font=row_font,
            fill=TEXT,
        )

        projected_total = float(row["proj_total"])
        _draw_market(
            draw,
            x=500,
            y=text_y,
            row=row.get("moneyline"),
            market="moneyline",
            projected_total=projected_total,
            row_font=row_font,
            bold_font=bold_font,
            small_font=small_font,
            badge_font=badge_font,
        )
        _draw_market(
            draw,
            x=790,
            y=text_y,
            row=row.get("spread"),
            market="spread",
            projected_total=projected_total,
            row_font=row_font,
            bold_font=bold_font,
            small_font=small_font,
            badge_font=badge_font,
        )
        _draw_market(
            draw,
            x=1075,
            y=text_y,
            row=row.get("total"),
            market="total",
            projected_total=projected_total,
            row_font=row_font,
            bold_font=bold_font,
            small_font=small_font,
            badge_font=badge_font,
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def write_weekly_publication(
    current: pl.DataFrame,
    *,
    week: int,
    updated_at: str,
    release_state: str,
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    board = build_weekly_board(current)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    html_path = directory / f"nfl_week_{week}.html"
    render_html(
        board,
        html_path,
        week=week,
        updated_at=updated_at,
        release_state=release_state,
    )

    pages = max(1, math.ceil(board.height / 14))
    png_paths: list[str] = []
    for page in range(1, pages + 1):
        image_path = directory / f"nfl_week_{week}_page{page}.png"
        render_png(
            board,
            image_path,
            page=page,
            week=week,
            updated_at=updated_at,
            release_state=release_state,
        )
        png_paths.append(str(image_path))

    return {
        "board_games": board.height,
        "html": str(html_path),
        "png_pages": png_paths,
        "release_state": release_state.upper(),
        "presentation": "cfb_style_weekly_picks_v1",
    }
