"""Human-readable NFL weekly publication modeled after the NCAA board."""

from __future__ import annotations

import html
import math
from pathlib import Path

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


def _market_cell(row: dict[str, object] | None, *, release_state: str) -> str:
    if not row:
        return "NO LINE"
    side = _team_side(row)
    market = str(row.get("quant_market") or "")
    line = "" if market == "moneyline" else f" {_fmt_line(row.get('quant_price'))}"
    odds = _fmt_odds(row.get("quant_odds"))
    signal = str(row.get("quant_signal") or "PASS").upper()
    book = str(row.get("quant_book") or "")
    action = str(row.get("portfolio_action") or "PASS").upper()
    label = signal if signal != "PASS" else ""
    if action in {"PAPER", "SHADOW"} and label:
        label = f"{label}/{action}"
    if action == "BET" and release_state == "PRODUCTION" and label:
        label = f"{label}/BET"
    suffix = f" · {label}" if label else ""
    book_text = f" · {book}" if book else ""
    return f"{side}{line} {odds}{book_text}{suffix}".strip()


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
        context_quality = _number(base.get("context_quality"))
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
                "moneyline": markets.get("moneyline"),
                "spread": markets.get("spread"),
                "total": markets.get("total"),
                "context_quality": context_quality,
            }
        )
    return pl.DataFrame(rows).sort(["date", "game_id"])


def _badge_html(label: str) -> str:
    if not label or label == "PASS":
        return ""
    css = label.split("/", 1)[0].lower()
    return f'<span class="badge {css}">{html.escape(label)}</span>'


def _market_html(row: dict[str, object] | None, *, release_state: str) -> str:
    if not row:
        return '<span class="quiet">NO LINE</span>'
    text = _market_cell(row, release_state=release_state)
    signal = str(row.get("quant_signal") or "PASS").upper()
    action = str(row.get("portfolio_action") or "PASS").upper()
    label = signal
    if signal != "PASS" and action in {"PAPER", "SHADOW"}:
        label = f"{signal}/{action}"
    elif signal != "PASS" and action == "BET" and release_state == "PRODUCTION":
        label = f"{signal}/BET"
    if label and label != "PASS":
        raw = text.rsplit(f" · {label}", 1)[0]
        return f'{html.escape(raw)} {_badge_html(label)}'
    return html.escape(text)


def render_html(
    board: pl.DataFrame,
    path: str | Path,
    *,
    week: int,
    updated_at: str,
    release_state: str,
) -> None:
    state = release_state.upper()
    state_note = (
        "Production eligible · execution still requires current verified quote provenance"
        if state == "PRODUCTION"
        else f"{state} evidence mode · displayed opportunities are not production staking"
    )
    css = """
*{box-sizing:border-box}body{margin:0;background:#0f1113;color:#f0f1f2;font-family:Inter,Arial,sans-serif}
.shell{max-width:1360px;margin:auto;padding:14px}.page{display:none}.head{display:flex;justify-content:space-between;align-items:flex-start}
.title{font-size:24px;font-weight:800}.sub,.counter{font-size:12px;color:#8b8e92}.status{font-size:10px;color:#d6a44b;margin-top:4px}
.counter{text-align:right;line-height:1.45}table{width:100%;border-collapse:collapse;table-layout:fixed;margin-top:8px}
th{font-size:9px;letter-spacing:.08em;color:#777c81;text-align:left;padding:7px 8px;border-bottom:1px solid #272a2e}
td{font-size:12px;padding:8px;border-bottom:1px solid #272a2e;height:38px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
tbody tr:nth-child(even){background:#16181b}.winner{font-weight:800}.at{color:#777c81}.proj{font-weight:800}.bar{display:inline-block;height:6px;background:#4c84e0;border-radius:5px;margin-right:6px;vertical-align:middle}
.quiet{color:#64686d}.badge{font-size:8px;font-weight:800;border-radius:4px;padding:3px 5px;margin-left:4px}.strong{background:#5fc468;color:#0c2c12}.bet{background:#1f462a;color:#63c76d}.lean{background:#483b1e;color:#e3b549}
.nav{text-align:center;padding:14px}.nav button{background:#202328;border:1px solid #373b40;color:#eee;border-radius:5px;padding:6px 10px;margin:2px}
.c1{width:18%}.c2{width:7%}.c3{width:8%}.c4{width:22%}.c5{width:22%}.c6{width:18%}.c7{width:5%}@media(max-width:900px){.shell{overflow-x:auto}.page{min-width:1180px}}
"""
    values = board.to_dicts()
    pages: list[str] = []
    page_count = max(1, math.ceil(len(values) / 14))
    for page in range(1, page_count + 1):
        start = (page - 1) * 14
        chunk = values[start : start + 14]
        table_rows: list[str] = []
        for row in chunk:
            away = html.escape(str(row["away_team"]))
            home = html.escape(str(row["home_team"]))
            winner = str(row["winner"])
            matchup = f'<span class="winner">{away}</span>' if row["away_team"] == winner else away
            matchup += ' <span class="at">@</span> '
            matchup += f'<span class="winner">{home}</span>' if row["home_team"] == winner else home
            pct = int(row["win_pct"])
            bar = max(4, min(52, int((pct - 50) * 1.15)))
            context = row.get("context_quality")
            context_text = "—" if context is None else f"{100 * float(context):.0f}%"
            table_rows.append(
                "<tr>"
                f"<td>{matchup}</td>"
                f"<td class=\"proj\">{row['away_score']}–{row['home_score']}</td>"
                f"<td><span class=\"bar\" style=\"width:{bar}px\"></span>{pct}%</td>"
                f"<td>{_market_html(row.get('moneyline'), release_state=state)}</td>"
                f"<td>{_market_html(row.get('spread'), release_state=state)}</td>"
                f"<td>{_market_html(row.get('total'), release_state=state)}</td>"
                f"<td>{context_text}</td>"
                "</tr>"
            )
        game_range = (
            f"Games {start + 1}–{min(start + 14, len(values))} of {len(values)}"
            if values
            else "0 games"
        )
        pages.append(
            f'<section class="page" id="p{page}">'
            '<div class="head"><div>'
            f'<div class="title">NFL MODEL · WEEK {week} BOARD</div>'
            f'<div class="sub">Projected scores &amp; line-shopped markets · Updated {html.escape(updated_at)}</div>'
            f'<div class="status">{html.escape(state_note)}</div></div>'
            f'<div class="counter">{game_range}<br>{page} / {page_count}</div></div>'
            '<table><colgroup><col class="c1"><col class="c2"><col class="c3">'
            '<col class="c4"><col class="c5"><col class="c6"><col class="c7"></colgroup>'
            '<thead><tr><th>MATCHUP</th><th>PROJ</th><th>WIN %</th><th>MONEYLINE</th>'
            '<th>SPREAD</th><th>TOTAL</th><th>CTX</th></tr></thead>'
            f'<tbody>{"".join(table_rows)}</tbody></table></section>'
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


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    font: ImageFont.ImageFont,
) -> int:
    if not label or label == "PASS":
        return 0
    base = label.split("/", 1)[0]
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0] + 12
    fill = GREEN if base == "STRONG" else DARK_GREEN if base == "BET" else AMBER
    text = (12, 45, 17) if base == "STRONG" else GREEN if base == "BET" else AMBER_TEXT
    draw.rounded_rectangle((x, y - 2, x + width, y + 13), radius=4, fill=fill)
    draw.text((x + 6, y + 1), label, font=font, fill=text)
    return width + 5


def render_png(
    board: pl.DataFrame,
    path: str | Path,
    *,
    page: int,
    week: int,
    updated_at: str,
    release_state: str,
) -> None:
    width, height = 1400, 700
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = _font(22, bold=True)
    sub_font = _font(11)
    status_font = _font(9, bold=True)
    head_font = _font(9, bold=True)
    row_font = _font(11)
    bold_font = _font(11, bold=True)
    small_font = _font(8)
    badge_font = _font(7, bold=True)
    state = release_state.upper()

    draw.text((17, 13), f"NFL MODEL · WEEK {week} BOARD", font=title_font, fill=TEXT)
    draw.text(
        (17, 43),
        f"Projected scores & line-shopped markets · Updated {updated_at}",
        font=sub_font,
        fill=MUTED,
    )
    state_note = (
        "PRODUCTION eligible"
        if state == "PRODUCTION"
        else f"{state} evidence mode · not production staking"
    )
    draw.text((17, 59), state_note, font=status_font, fill=WARN)

    values = board.to_dicts()
    page_count = max(1, math.ceil(len(values) / 14))
    page = max(1, min(page, page_count))
    start = (page - 1) * 14
    end = min(start + 14, len(values))
    draw.text(
        (1210, 17),
        f"Games {start + 1 if values else 0}–{end} of {len(values)}",
        font=sub_font,
        fill=MUTED,
    )
    draw.text((1330, 39), f"{page} / {page_count}", font=sub_font, fill=MUTED)

    columns = [17, 255, 335, 430, 720, 1010, 1320]
    headings = ["MATCHUP", "PROJ", "WIN %", "MONEYLINE", "SPREAD", "TOTAL", "CTX"]
    for x, heading in zip(columns, headings, strict=True):
        draw.text((x, 78), heading, font=head_font, fill=(119, 124, 129))
    draw.line((16, 96, 1384, 96), fill=GRID)

    for index, row in enumerate(values[start:end]):
        y = 97 + index * 41
        if index % 2:
            draw.rectangle((16, y, 1384, y + 40), fill=ALT)
        draw.line((16, y, 1384, y), fill=GRID)
        text_y = y + 13
        away = str(row["away_team"])
        home = str(row["home_team"])
        winner = str(row["winner"])
        away_font = bold_font if away == winner else row_font
        home_font = bold_font if home == winner else row_font
        draw.text((17, text_y), away, font=away_font, fill=TEXT)
        away_width = draw.textbbox((0, 0), away, font=away_font)[2]
        draw.text((17 + away_width, text_y), " @ ", font=row_font, fill=MUTED)
        at_width = draw.textbbox((0, 0), " @ ", font=row_font)[2]
        draw.text((17 + away_width + at_width, text_y), home, font=home_font, fill=TEXT)
        draw.text(
            (255, text_y),
            f"{row['away_score']}–{row['home_score']}",
            font=bold_font,
            fill=TEXT,
        )
        pct = int(row["win_pct"])
        bar_width = max(4, min(46, int((pct - 50) * 1.05)))
        draw.rounded_rectangle(
            (335, text_y + 6, 335 + bar_width, text_y + 11),
            radius=3,
            fill=BLUE,
        )
        draw.text((335 + bar_width + 7, text_y - 1), f"{pct}%", font=row_font, fill=TEXT)

        for x, key in ((430, "moneyline"), (720, "spread"), (1010, "total")):
            market = row.get(key)
            if not market:
                draw.text((x, text_y), "NO LINE", font=small_font, fill=(96, 100, 105))
                continue
            market_text = _market_cell(market, release_state=state)
            label = str(market.get("quant_signal") or "PASS").upper()
            action = str(market.get("portfolio_action") or "PASS").upper()
            if label != "PASS" and action in {"PAPER", "SHADOW"}:
                label = f"{label}/{action}"
            elif label != "PASS" and action == "BET" and state == "PRODUCTION":
                label = f"{label}/BET"
            if label != "PASS":
                market_text = market_text.rsplit(f" · {label}", 1)[0]
            display = market_text[:34]
            active = label != "PASS"
            font = bold_font if active else row_font
            draw.text((x, text_y), display, font=font, fill=TEXT if active else MUTED)
            if active:
                text_width = draw.textbbox((0, 0), display, font=font)[2]
                _draw_badge(draw, x + text_width + 5, text_y, label, badge_font)

        context = row.get("context_quality")
        context_text = "—" if context is None else f"{100 * float(context):.0f}%"
        draw.text((1320, text_y), context_text, font=row_font, fill=MUTED)

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
        path = directory / f"nfl_week_{week}_page{page}.png"
        render_png(
            board,
            path,
            page=page,
            week=week,
            updated_at=updated_at,
            release_state=release_state,
        )
        png_paths.append(str(path))
    return {
        "board_games": board.height,
        "html": str(html_path),
        "png_pages": png_paths,
        "release_state": release_state.upper(),
    }
