"""NCAA-style publication bundle and reconciliation for canonical NFL runs."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .policy import load_policy

PUBLICATION_SCHEMA_VERSION = 1


def _read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        value = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _number(value: object, default: float | None = None) -> float | None:
    return float(value) if _finite(value) else default


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = "ERROR",
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    }


def _json_equal(first: object, second: object) -> bool:
    return json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second,
        sort_keys=True,
        default=str,
    )


def _load_trend(path: str | Path, *, limit: int = 12) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, object]] = []
    try:
        lines = source.read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows[-max(1, int(limit)) :]


def _compact_evidence(report: dict[str, object]) -> dict[str, object]:
    overall = report.get("overall")
    if not isinstance(overall, dict):
        overall = {}
    return {
        "status": report.get("status", "UNKNOWN"),
        "bets": int(overall.get("bets", 0) or 0),
        "roi": _number(overall.get("roi", overall.get("roi_per_unit_staked"))),
        "units": _number(overall.get("units", overall.get("net_units")), 0.0),
        "avg_clv": _number(overall.get("avg_clv", overall.get("average_clv_proxy"))),
        "roi_ci_95": overall.get("roi_ci_95")
        or [overall.get("roi_ci_95_low"), overall.get("roi_ci_95_high")],
        "max_drawdown": _number(overall.get("max_drawdown"), 0.0),
    }


def _compact_live(report: dict[str, object]) -> dict[str, object]:
    overall = report.get("overall")
    if not isinstance(overall, dict):
        overall = {}
    return {
        "status": report.get("status", "UNKNOWN"),
        "portfolio_verified": bool(report.get("portfolio_verified", False)),
        "graded_bets": int(report.get("graded_bets", overall.get("bets", 0)) or 0),
        "roi": _number(report.get("roi", overall.get("roi"))),
        "units": _number(overall.get("units"), 0.0),
        "avg_clv": _number(report.get("avg_clv", overall.get("avg_clv"))),
        "roi_ci_95": overall.get("roi_ci_95") or [None, None],
    }


def build_publication_snapshot(
    current: pl.DataFrame,
    report: dict[str, object],
    *,
    live: dict[str, object] | None = None,
    prior_trend: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one reconciled view of the machine report and public NFL publication."""

    meta = report.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    gate = report.get("release_gate")
    if not isinstance(gate, dict):
        gate = {}
    monitor = report.get("monitoring")
    if not isinstance(monitor, dict):
        monitor = {}
    portfolio = report.get("portfolio")
    if not isinstance(portfolio, dict):
        portfolio = {}
    health = report.get("health")
    if not isinstance(health, dict):
        health = {}
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    data_quality = meta.get("data_quality")
    if not isinstance(data_quality, dict):
        data_quality = {}
    market = meta.get("market_coverage")
    if not isinstance(market, dict):
        market = {}
    intel = meta.get("market_intelligence")
    if not isinstance(intel, dict):
        intel = {}
    context = meta.get("current_context")
    if not isinstance(context, dict):
        context = {}
    probability = meta.get("probability")
    if not isinstance(probability, dict):
        probability = {}
    policy = load_policy()

    checks: list[dict[str, object]] = []
    expected_market_rows = int(data_quality.get("market_rows", current.height) or 0)
    checks.append(
        _check(
            "market_row_count",
            current.height == expected_market_rows,
            f"predictions={current.height} metadata={expected_market_rows}",
        )
    )
    game_count = current.get_column("game_id").n_unique() if "game_id" in current.columns else 0
    expected_games = int(market.get("games", game_count) or 0)
    checks.append(
        _check(
            "game_count",
            game_count == expected_games,
            f"prediction_games={game_count} metadata={expected_games}",
        )
    )
    for key in ("season", "week"):
        if key not in current.columns or current.is_empty():
            checks.append(_check(f"identity_{key}", False, f"missing live {key}"))
            continue
        values = sorted(
            {int(value) for value in current.get_column(key).drop_nulls().unique().to_list()}
        )
        target = int(meta[key]) if meta.get(key) is not None else None
        checks.append(
            _check(
                f"identity_{key}",
                target is not None and values == [target],
                f"values={values} metadata={target}",
            )
        )

    top_state = str(report.get("release_state", "UNKNOWN"))
    gate_state = str(gate.get("release_state", "UNKNOWN"))
    checks.append(
        _check(
            "release_gate_reconciled",
            top_state == gate_state,
            f"report={top_state} gate={gate_state}",
        )
    )
    checks.append(
        _check(
            "data_contracts_publishable",
            str(data_quality.get("status", "FAIL")).upper() != "FAIL",
            f"status={data_quality.get('status', 'UNKNOWN')}",
        )
    )

    approved = float(portfolio.get("approved_units", 0.0) or 0.0)
    portfolio_mode = str(portfolio.get("mode", "paper") or "paper").lower()
    policy_mode = str(policy.get("deployment_mode", "paper") or "paper").lower()
    production_safe = (
        bool(gate.get("production_eligible", False))
        and gate_state == "PRODUCTION"
        and policy_mode == "production"
        and portfolio_mode == "production"
    )
    checks.append(
        _check(
            "approved_stake_fail_closed",
            approved <= 1e-9 or production_safe,
            (
                f"approved={approved:.3f} release={gate_state} "
                f"policy={policy_mode} portfolio={portfolio_mode}"
            ),
        )
    )
    allocated = portfolio.get("paper_or_shadow_allocated_units")
    if _finite(allocated):
        checks.append(
            _check(
                "approved_not_above_allocation",
                approved <= float(allocated) + 1e-9,
                f"approved={approved:.3f} allocated={float(allocated):.3f}",
            )
        )

    publication = report.get("publication")
    if not isinstance(publication, dict):
        publication = {}
    html_path = Path(str(publication.get("html") or ""))
    png_paths = [Path(str(value)) for value in publication.get("png_pages", [])]
    checks.append(
        _check(
            "weekly_html_present",
            bool(str(publication.get("html") or "")) and html_path.exists(),
            f"path={publication.get('html')}",
        )
    )
    checks.append(
        _check(
            "weekly_png_present",
            bool(png_paths) and all(path.exists() for path in png_paths),
            f"pages={len(png_paths)}",
        )
    )

    prior = prior_trend or []
    readiness = _number(monitor.get("live_readiness_score"), 0.0) or 0.0
    prior_scores = [
        float(row["live_readiness_score"])
        for row in prior[-5:]
        if _finite(row.get("live_readiness_score"))
    ]
    trend_delta = None
    if prior_scores:
        ordered = sorted(prior_scores)
        middle = len(ordered) // 2
        baseline = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        trend_delta = readiness - baseline

    errors = [
        check
        for check in checks
        if check["severity"] == "ERROR" and not bool(check["passed"])
    ]
    warnings = list(
        dict.fromkeys(
            [str(value) for value in monitor.get("alerts", [])]
            + [str(value) for value in gate.get("blockers", [])]
        )
    )
    if trend_delta is not None and trend_delta <= -15:
        warnings.append(
            f"live readiness dropped {abs(trend_delta):.1f} points versus recent median"
        )
    status = (
        "FAIL"
        if errors
        else "WARN"
        if warnings or str(data_quality.get("status", "")).upper() == "WARN"
        else "PASS"
    )

    live_report = live or {}
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "identity": {
            "league": "NFL",
            "season": meta.get("season"),
            "week": meta.get("week"),
            "model_generated_at": meta.get("generated_at"),
            "prediction_rows": current.height,
            "prediction_games": game_count,
        },
        "release": {
            "state": gate_state,
            "production_eligible": bool(gate.get("production_eligible", False)),
            "engineering_ready": bool(gate.get("engineering_ready", False)),
            "historical_edge_ready": bool(gate.get("historical_edge_ready", False)),
            "live_evidence_ready": bool(gate.get("live_evidence_ready", False)),
            "checks": gate.get("checks", []),
            "blockers": gate.get("blockers", []),
            "next_requirements": gate.get("next_requirements", []),
        },
        "probability": probability,
        "coverage": {
            "games": expected_games,
            "moneyline": int(market.get("moneyline", 0) or 0),
            "spread": int(market.get("spread", 0) or 0),
            "total": int(market.get("total", 0) or 0),
            "context": _number(context.get("coverage"), 0.0),
            "quarterback": _number(context.get("qb_coverage"), 0.0),
            "multi_book": _number(intel.get("multi_book_coverage"), 0.0),
        },
        "monitoring": {
            "status": monitor.get("status", "UNKNOWN"),
            "live_readiness_score": readiness,
            "scores": monitor.get("scores", {}),
            "drift_details": monitor.get("drift_details", {}),
            "alerts": monitor.get("alerts", []),
            "recent_readiness_delta": trend_delta,
        },
        "data_quality": data_quality,
        "health": health,
        "portfolio": portfolio,
        "historical_evidence": _compact_evidence(evidence),
        "live_evidence": _compact_live(live_report),
        "reconciliation": {
            "status": "FAIL" if errors else "PASS",
            "checks": checks,
            "errors": errors,
        },
        "trend": prior[-12:],
        "warnings": warnings,
        "meaning": (
            "NFL publication/audit snapshot. Readiness, model error, historical evidence, "
            "and forward evidence remain separate; none guarantees future profitability."
        ),
    }


def _trend_record(snapshot: dict[str, object]) -> dict[str, object]:
    identity = snapshot.get("identity")
    if not isinstance(identity, dict):
        identity = {}
    monitoring = snapshot.get("monitoring")
    if not isinstance(monitoring, dict):
        monitoring = {}
    portfolio = snapshot.get("portfolio")
    if not isinstance(portfolio, dict):
        portfolio = {}
    release = snapshot.get("release")
    if not isinstance(release, dict):
        release = {}
    quality = snapshot.get("data_quality")
    if not isinstance(quality, dict):
        quality = {}
    scores = monitoring.get("scores")
    if not isinstance(scores, dict):
        scores = {}
    return {
        "generated_at": snapshot.get("generated_at"),
        "model_generated_at": identity.get("model_generated_at"),
        "season": identity.get("season"),
        "week": identity.get("week"),
        "release_state": release.get("state"),
        "publication_status": snapshot.get("status"),
        "live_readiness_score": monitoring.get("live_readiness_score"),
        "distribution_stability": scores.get("distribution_stability"),
        "approved_units": portfolio.get("approved_units", 0.0),
        "data_quality_status": quality.get("status"),
    }


def _append_trend(snapshot: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = _trend_record(snapshot)
    existing = _load_trend(target, limit=200)
    signature = (
        record.get("model_generated_at"),
        record.get("season"),
        record.get("week"),
    )
    if existing:
        last = existing[-1]
        last_signature = (
            last.get("model_generated_at"),
            last.get("season"),
            last.get("week"),
        )
        if signature == last_signature:
            return
    with target.open("a") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")


def _percent(value: object) -> str:
    number = _number(value)
    return "—" if number is None else f"{100.0 * number:.2f}%"


def _fmt(value: object, digits: int = 3) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def render_run_report(snapshot: dict[str, object]) -> str:
    identity = snapshot.get("identity") or {}
    release = snapshot.get("release") or {}
    probability = snapshot.get("probability") or {}
    monitoring = snapshot.get("monitoring") or {}
    portfolio = snapshot.get("portfolio") or {}
    historical = snapshot.get("historical_evidence") or {}
    live = snapshot.get("live_evidence") or {}
    reconciliation = snapshot.get("reconciliation") or {}
    blockers = release.get("blockers", []) if isinstance(release, dict) else []
    alerts = monitoring.get("alerts", []) if isinstance(monitoring, dict) else []
    lines = [
        "# Harbin NFL Run Report",
        "",
        f"**Season / Week:** {identity.get('season')} / {identity.get('week')}  ",
        f"**Publication status:** {snapshot.get('status')}  ",
        f"**Release state:** {release.get('state')}  ",
        f"**Reconciliation:** {reconciliation.get('status')}  ",
        "",
        "## Probability validation",
        f"- Home-win Brier: **{_fmt(probability.get('home_win_brier'), 4)}**.",
        (
            "- Margin / total 80% coverage: "
            f"**{_percent(probability.get('margin_80_coverage'))} / "
            f"{_percent(probability.get('total_80_coverage'))}**."
        ),
        "",
        "## Monitoring and execution",
        (
            "- Live readiness: "
            f"**{_fmt(monitoring.get('live_readiness_score'), 1)}/100**."
        ),
        (
            f"- Portfolio mode: **{str(portfolio.get('mode', 'paper')).upper()}**; "
            f"proposed **{_fmt(portfolio.get('proposed_units'), 2)}u**; "
            f"approved **{_fmt(portfolio.get('approved_units'), 2)}u**."
        ),
        (
            f"- Historical evidence: **{historical.get('bets', 0)} bets**, "
            f"ROI **{_percent(historical.get('roi'))}**, "
            f"CLV **{_percent(historical.get('avg_clv'))}**."
        ),
        (
            f"- Independent forward evidence: **{live.get('graded_bets', 0)} bets**, "
            f"ROI **{_percent(live.get('roi'))}**, "
            f"CLV **{_percent(live.get('avg_clv'))}**."
        ),
        "",
        "## Current blockers",
    ]
    lines.extend([f"- {value}" for value in blockers] or ["- None."])
    lines.extend(["", "## Operational alerts"])
    lines.extend([f"- {value}" for value in alerts] or ["- None."])
    lines.extend(
        [
            "",
            "## Interpretation",
            (
                "A green software run, high readiness score, or low model error does not "
                "establish a profitable betting edge. Historical and independently graded "
                "forward evidence remain separate release requirements."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_model_card_markdown(card: dict[str, object]) -> str:
    core = card.get("core_model") if isinstance(card.get("core_model"), dict) else {}
    probability = card.get("probability") if isinstance(card.get("probability"), dict) else {}
    markets = card.get("markets") if isinstance(card.get("markets"), dict) else {}
    risk = card.get("risk") if isinstance(card.get("risk"), dict) else {}
    rules = card.get("non_negotiables") if isinstance(card.get("non_negotiables"), list) else []
    lines = [
        "# Harbin NFL Analytics — Model Card",
        "",
        f"**Release state:** {card.get('release_state', 'UNKNOWN')}  ",
        f"**Production eligible:** {card.get('production_eligible', False)}  ",
        "",
        "## Core model",
        f"- Fair score: {core.get('fair_score', 'unknown')}",
        f"- Sportsbook prices in score model: {core.get('sportsbook_prices_in_score_model')}",
        f"- Prior-season weight: {core.get('prior_season_weight')}",
        f"- Quarterback layer: {core.get('quarterback_layer')}",
        "",
        "## Probability and markets",
        f"- Probability method: {probability.get('method', 'unknown')}",
        f"- Historical market source: {markets.get('historical_primary', 'unknown')}",
        f"- Current market source: {markets.get('current_primary', 'unknown')}",
        f"- Optional enrichment: {markets.get('optional_enrichment', 'none')}",
        "",
        "## Risk controls",
        f"- Staking: {risk.get('staking', 'unknown')}",
        f"- Portfolio caps: {risk.get('portfolio_caps', False)}",
        f"- Drawdown throttle: {risk.get('drawdown_throttle', False)}",
        "",
        "## Non-negotiables",
    ]
    lines.extend([f"- {value}" for value in rules])
    lines.append("")
    return "\n".join(lines)


def _card_columns(current: pl.DataFrame) -> list[str]:
    preferred = [
        "season",
        "week",
        "game_id",
        "date",
        "kickoff",
        "away_team",
        "home_team",
        "model_margin_home",
        "model_total",
        "quant_signal",
        "quant_market",
        "quant_side",
        "quant_book",
        "quant_price",
        "quant_odds",
        "quant_probability",
        "quant_edge",
        "quant_ev",
        "market_book_count",
        "portfolio_candidate_units",
        "portfolio_stake_units",
        "portfolio_action",
        "execution_ready",
    ]
    return [column for column in preferred if column in current.columns]


def _write_cards(current: pl.DataFrame, *, output_dir: Path, docs_dir: Path) -> None:
    columns = _card_columns(current)
    card = current.select(columns) if columns and not current.is_empty() else pl.DataFrame()
    if not card.is_empty():
        card.write_csv(output_dir / "portfolio_card.csv")
    recommendations = (
        card.filter(pl.col("quant_signal") != "PASS")
        if not card.is_empty() and "quant_signal" in card.columns
        else pl.DataFrame()
    )
    if not recommendations.is_empty():
        recommendations.write_csv(output_dir / "quant_recommendations.csv")

    rows: list[str] = []
    for row in recommendations.to_dicts():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('away_team', '')))} @ "
            f"{html.escape(str(row.get('home_team', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_signal', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_market', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_side', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_price', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_odds', '')))}</td>"
            f"<td>{html.escape(str(row.get('quant_book', '')))}</td>"
            f"<td>{html.escape(str(row.get('portfolio_action', '')))}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="8">No non-PASS opportunities.</td></tr>'
    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Harbin NFL Quant Card</title><style>"
        "body{background:#0f1113;color:#f0f1f2;font-family:Arial,sans-serif;margin:24px}"
        "table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #333}"
        "th{text-align:left;color:#969ba1}a{color:#74a7ff}</style></head><body>"
        "<h1>Harbin NFL · Quant Card</h1>"
        "<p>Research/publication view. Portfolio action and release gate control execution.</p>"
        "<p><a href=\"./\">← Weekly board</a> · <a href=\"audit.html\">System audit</a></p>"
        "<table><thead><tr><th>Matchup</th><th>Signal</th><th>Market</th><th>Side</th>"
        "<th>Line</th><th>Odds</th><th>Book</th><th>Action</th></tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
    (output_dir / "quant_card.html").write_text(document)
    (docs_dir / "quant.html").write_text(document)


AUDIT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harbin NFL Audit</title><style>
body{margin:24px;background:#0f1113;color:#f3f4f5;font-family:Arial,sans-serif}
a{color:#74a7ff}.card{background:#171a1e;border:1px solid #292d32;border-radius:10px;padding:16px;margin:12px 0}
pre{white-space:pre-wrap}</style></head><body><h1>Harbin NFL · System Audit</h1>
<p>One reconciled publication snapshot. Readiness is not a profitability guarantee.</p>
<p><a href="./">← Weekly board</a> · <a href="run_report.md">Run report</a> · <a href="quant.html">Quant card</a></p>
<div id="status" class="card">Loading audit snapshot…</div><div id="detail" class="card"></div>
<script>fetch('audit_snapshot.json?t='+Date.now()).then(r=>r.json()).then(s=>{
document.getElementById('status').innerHTML='<b>Publication:</b> '+s.status+' · <b>Release:</b> '+(s.release||{}).state+' · <b>Readiness:</b> '+((s.monitoring||{}).live_readiness_score??'—')+'/100';
document.getElementById('detail').innerHTML='<pre>'+JSON.stringify({reconciliation:s.reconciliation,warnings:s.warnings,coverage:s.coverage,historical_evidence:s.historical_evidence,live_evidence:s.live_evidence},null,2)+'</pre>';}).catch(()=>{document.getElementById('status').textContent='No audit snapshot has been published.'});</script></body></html>"""


def validate_publication_files(
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
) -> dict[str, object]:
    """Fail closed when public docs disagree with canonical NFL output state."""

    outputs = Path(output_dir)
    docs = Path(docs_dir)
    snapshot = _read_json(outputs / "audit_snapshot.json")
    public_snapshot = _read_json(docs / "audit_snapshot.json")
    current_model = _read_json(outputs / "current_model.json")
    latest = _read_json(docs / "latest.json")
    metadata = _read_json(docs / "metadata.json")
    gate = _read_json(docs / "release_gate.json")
    portfolio = _read_json(docs / "portfolio_summary.json")
    monitor = _read_json(docs / "live_monitoring.json")
    quality = _read_json(docs / "data_quality.json")

    checks: list[dict[str, object]] = []
    checks.append(
        _check("audit_snapshot_present", bool(snapshot), "outputs/audit_snapshot.json parses")
    )
    checks.append(
        _check("public_snapshot_present", bool(public_snapshot), "docs/audit_snapshot.json parses")
    )
    if snapshot and public_snapshot:
        checks.append(
            _check(
                "snapshot_copy_exact",
                _json_equal(snapshot, public_snapshot),
                "outputs and docs audit snapshots are identical",
            )
        )
    checks.append(
        _check("latest_model_present", bool(latest), "docs/latest.json parses")
    )
    if current_model and latest:
        checks.append(
            _check(
                "latest_model_copy_exact",
                _json_equal(current_model, latest),
                "outputs/current_model.json and docs/latest.json are identical",
            )
        )

    identity = snapshot.get("identity") if isinstance(snapshot.get("identity"), dict) else {}
    if metadata:
        checks.append(
            _check(
                "metadata_season",
                metadata.get("season") == identity.get("season"),
                f"docs={metadata.get('season')} snapshot={identity.get('season')}",
            )
        )
        checks.append(
            _check(
                "metadata_week",
                metadata.get("week") == identity.get("week"),
                f"docs={metadata.get('week')} snapshot={identity.get('week')}",
            )
        )
    else:
        checks.append(_check("metadata_present", False, "docs/metadata.json missing or invalid"))

    release = snapshot.get("release") if isinstance(snapshot.get("release"), dict) else {}
    checks.append(
        _check(
            "release_copy",
            bool(gate) and gate.get("release_state") == release.get("state"),
            f"docs={gate.get('release_state')} snapshot={release.get('state')}",
        )
    )
    snapshot_portfolio = (
        snapshot.get("portfolio") if isinstance(snapshot.get("portfolio"), dict) else {}
    )
    checks.append(
        _check(
            "portfolio_copy",
            bool(portfolio)
            and _number(portfolio.get("approved_units"), 0.0)
            == _number(snapshot_portfolio.get("approved_units"), 0.0),
            (
                f"docs={portfolio.get('approved_units')} "
                f"snapshot={snapshot_portfolio.get('approved_units')}"
            ),
        )
    )
    snapshot_monitor = (
        snapshot.get("monitoring") if isinstance(snapshot.get("monitoring"), dict) else {}
    )
    checks.append(
        _check(
            "monitor_copy",
            bool(monitor)
            and _number(monitor.get("live_readiness_score"), -999.0)
            == _number(snapshot_monitor.get("live_readiness_score"), -998.0),
            (
                f"docs={monitor.get('live_readiness_score')} "
                f"snapshot={snapshot_monitor.get('live_readiness_score')}"
            ),
        )
    )
    checks.append(
        _check("data_quality_present", bool(quality), "docs/data_quality.json parses")
    )
    checks.append(_check("public_index_present", (docs / "index.html").exists(), "docs/index.html exists"))
    checks.append(_check("public_quant_present", (docs / "quant.html").exists(), "docs/quant.html exists"))
    checks.append(_check("public_audit_present", (docs / "audit.html").exists(), "docs/audit.html exists"))

    errors = [
        check
        for check in checks
        if check["severity"] == "ERROR" and not bool(check["passed"])
    ]
    return {
        "status": "FAIL" if errors else "PASS",
        "checks": checks,
        "errors": errors,
    }


def write_publication_bundle(
    current: pl.DataFrame,
    report: dict[str, object],
    *,
    output_dir: str | Path = "outputs",
    docs_dir: str | Path = "docs",
    reports_dir: str | Path = "reports",
    history_dir: str | Path = "history",
) -> dict[str, object]:
    """Publish the NCAA-style NFL bundle, then reconcile every public state copy."""

    outputs = Path(output_dir)
    docs = Path(docs_dir)
    reports = Path(reports_dir)
    history = Path(history_dir)
    for directory in (outputs, docs, reports, history):
        directory.mkdir(parents=True, exist_ok=True)

    live = _read_json(reports / "live_performance.json")
    trend_path = history / "audit_snapshots_v1.jsonl"
    prior = _load_trend(trend_path, limit=12)
    snapshot = build_publication_snapshot(current, report, live=live, prior_trend=prior)
    run_report = render_run_report(snapshot)

    for target in (outputs / "audit_snapshot.json", docs / "audit_snapshot.json"):
        target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str))
    (outputs / "RUN_REPORT.md").write_text(run_report)
    (docs / "run_report.md").write_text(run_report)
    (docs / "audit.html").write_text(AUDIT_HTML)
    (docs / ".nojekyll").touch()

    model_card = report.get("model_card")
    if not isinstance(model_card, dict):
        model_card = {}
    (outputs / "MODEL_CARD.md").write_text(render_model_card_markdown(model_card))
    (outputs / "README.md").write_text(
        "# Harbin NFL generated outputs\n\n"
        "Canonical generated artifacts. Read `RUN_REPORT.md`, `MODEL_CARD.md`, and "
        "`audit_snapshot.json` together; no single metric establishes a betting edge.\n"
    )

    meta = report.get("meta") if isinstance(report.get("meta"), dict) else {}
    gate = report.get("release_gate") if isinstance(report.get("release_gate"), dict) else {}
    monitor = report.get("monitoring") if isinstance(report.get("monitoring"), dict) else {}
    portfolio = report.get("portfolio") if isinstance(report.get("portfolio"), dict) else {}
    evidence = report.get("evidence") if isinstance(report.get("evidence"), dict) else {}
    health = report.get("health") if isinstance(report.get("health"), dict) else {}
    quality = meta.get("data_quality") if isinstance(meta.get("data_quality"), dict) else {}
    copies: dict[str, Any] = {
        "metadata.json": meta,
        "release_gate.json": gate,
        "live_monitoring.json": monitor,
        "portfolio_summary.json": portfolio,
        "evidence.json": evidence,
        "system_health.json": health,
        "data_quality.json": quality,
        "policy.json": load_policy(),
    }
    for filename, value in copies.items():
        (docs / filename).write_text(json.dumps(value, indent=2, sort_keys=True, default=str))

    publication = report.get("publication")
    if not isinstance(publication, dict):
        publication = {}
    weekly_html = Path(str(publication.get("html") or ""))
    if not weekly_html.exists():
        raise RuntimeError("weekly NFL HTML publication is missing")
    (docs / "index.html").write_text(weekly_html.read_text())

    current_model_path = outputs / "current_model.json"
    if not current_model_path.exists():
        raise RuntimeError("canonical outputs/current_model.json is missing before publication")
    (docs / "latest.json").write_text(current_model_path.read_text())
    current_csv = outputs / "current_predictions.csv"
    if current_csv.exists():
        (docs / "latest.csv").write_text(current_csv.read_text())

    _write_cards(current, output_dir=outputs, docs_dir=docs)
    _append_trend(snapshot, trend_path)

    validation = validate_publication_files(output_dir=outputs, docs_dir=docs)
    for target in (
        outputs / "publication_validation.json",
        docs / "publication_validation.json",
    ):
        target.write_text(json.dumps(validation, indent=2, sort_keys=True, default=str))
    if validation["status"] == "FAIL":
        details = "; ".join(str(item["detail"]) for item in validation["errors"])
        raise RuntimeError(f"NFL publication reconciliation failed: {details}")
    return {
        "status": validation["status"],
        "audit_snapshot": str(outputs / "audit_snapshot.json"),
        "run_report": str(outputs / "RUN_REPORT.md"),
        "model_card_markdown": str(outputs / "MODEL_CARD.md"),
        "portfolio_card": str(outputs / "portfolio_card.csv"),
        "quant_card": str(outputs / "quant_card.html"),
        "public_index": str(docs / "index.html"),
        "public_audit": str(docs / "audit.html"),
        "public_quant": str(docs / "quant.html"),
        "validation": str(outputs / "publication_validation.json"),
        "trend_history": str(trend_path),
    }
