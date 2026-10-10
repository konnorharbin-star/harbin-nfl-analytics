"""Prespecified all-game moneyline experiment; never authorizes a wager.

Market power is a pricing-layer hypothesis. Core football forecasts stay independent.
First-seen forecasts are immutable; no tuning or retrospective forward reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.grade_recommendations import fetch, final_result
from scripts.recommendation_ledger import timestamp

SPEC = "market_first_moneyline_v1"
EVALUATION_END = datetime(2027, 3, 1, tzinfo=UTC)
VARIANTS = ("independent_model", "market_model_25pct", "market_power_1_1")
MIN_GAMES, MIN_WEEKS = 100, 8


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def probability(value):
    x = number(value)
    return x if x is not None and 0 < x < 1 else None


def decimal(value):
    x = number(value)
    if x is None or not x.is_integer() or abs(x) < 100 or abs(x) > 10000:
        return None
    return 1 + (x / 100 if x > 0 else 100 / -x)


def forecasts(market, model):
    if probability(market) is None or probability(model) is None:
        raise ValueError("Missing valid paired market or independent model probability")
    # Symmetric power hypothesis: 1.1 log-odds, no fitted coefficient or ROI search.
    sharpened = 1 / (1 + math.exp(-1.1 * math.log(market / (1 - market))))
    return {
        "market": market,
        "independent_model": model,
        "market_model_25pct": 0.75 * market + 0.25 * model,
        "market_power_1_1": sharpened,
    }


def immutable(path, record):
    """Atomic creation; existing evidence can never be silently replaced."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        temp.write(payload)
        temp.flush()
        os.fsync(temp.fileno())
        name = temp.name
    try:
        os.link(name, path)
    except FileExistsError:
        return False
    finally:
        os.unlink(name)
    return True


def scores(rows, *, intervals=True):
    """One game = one observation. Resample full season/week clusters jointly."""
    valid = [
        r
        for r in rows
        if r.get("outcome") in (0, 1)
        and all(
            probability(r.get("predictions", {}).get(k)) is not None for k in ("market", *VARIANTS)
        )
    ]
    keys = [r["game_id"] for r in valid]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate game observations would inflate evidence")
    groups = defaultdict(list)
    for i, r in enumerate(valid):
        groups[(int(r["season"]), int(r["week"]))].append(i)
    result = {"games": len(valid), "week_clusters": len(groups), "variants": {}}
    if not valid:
        return result
    y = np.asarray([r["outcome"] for r in valid])
    loss = {}
    for variant in ("market", *VARIANTS):
        p = np.asarray([r["predictions"][variant] for r in valid])
        brier = (p - y) ** 2
        ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        ece = sum(
            abs(float(np.sum(p[(p * 10).astype(int) == b] - y[(p * 10).astype(int) == b])))
            for b in range(10)
        ) / len(y)
        result["variants"][variant] = {
            "brier": float(brier.mean()),
            "log_loss": float(ll.mean()),
            "ece_10": ece if len(y) >= MIN_GAMES else None,
        }
        loss[variant] = brier, ll
    supported = intervals and len(valid) >= MIN_GAMES and len(groups) >= MIN_WEEKS
    # 3 variants x 2 endpoints: Bonferroni intervals across six prespecified tests.
    tail = 0.05 / (2 * len(VARIANTS) * 2)
    if supported:
        rng = np.random.default_rng(20261010)
        blocks = list(groups.values())
        sample = rng.integers(0, len(blocks), (10000, len(blocks)))
        counts = np.array([len(x) for x in blocks])
    for variant in VARIANTS:
        record = result["variants"][variant]
        for i, metric in enumerate(("brier", "log_loss")):
            diff = loss["market"][i] - loss[variant][i]
            record[f"market_minus_candidate_{metric}"] = float(diff.mean())
            ci = None
            if supported:
                sums = np.array([diff[x].sum() for x in blocks])
                estimates = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
                ci = [float(x) for x in np.quantile(estimates, [tail, 1 - tail])]
            record[f"paired_familywise_95_{metric}_ci"] = ci
        record["interval_supported"] = supported
    return result


def nfl_history(path):
    result = []
    for r in csv.DictReader(path.open()):
        if r.get("market_type") != "moneyline" or r.get("result") not in ("win", "loss"):
            continue
        market, model = (
            probability(r.get("no_vig_probability")),
            probability(r.get("model_probability")),
        )
        if market is None or model is None:
            continue
        if r.get("side") not in ("home", "away"):
            raise ValueError("Unknown historical moneyline orientation")
        y = int(r["result"] == "win")
        if r["side"] == "away":
            market, model, y = 1 - market, 1 - model, 1 - y
        result.append(
            {
                "game_id": r["game_id"],
                "season": int(r["season"]),
                "week": int(r["week"]),
                "outcome": y,
                "predictions": forecasts(market, model),
            }
        )
    return result


def capture(sport, root, now, *, refresh_market=False):
    source = Path("docs/latest.csv")
    audit = json.loads(Path("docs/audit_snapshot.json").read_text())
    generated = timestamp(audit["generated_at"])
    if generated > now or now - generated > timedelta(hours=6):
        raise ValueError("Prediction publication is stale or future-dated")
    rows = list(csv.DictReader(source.open()))
    by_game = defaultdict(list)
    for row in rows:
        by_game[row["game_id"]].append(row)
    quotes = defaultdict(list)
    history_path = Path("history/market_snapshots.csv")
    if sport == "nfl":
        for q in csv.DictReader(history_path.open()):
            if q.get("market_type") != "moneyline":
                continue
            sides = {
                q["first_side"]: q["first_american_odds"],
                q["second_side"]: q["second_american_odds"],
            }
            quotes[q["game_id"]].append(
                {
                    "home_ml": sides.get("home"),
                    "away_ml": sides.get("away"),
                    "provider": q.get("book"),
                    "captured_at": q.get("captured_at"),
                    "source": q.get("provider"),
                    "source_event_id": q.get("source_event_id"),
                }
            )
    source_errors = []
    if refresh_market and sport == "nfl":
        import polars as pl

        from nfl.espn_market import ESPNMarketClient

        targets = [
            items[0]
            for items in by_game.values()
            if timestamp(items[0].get("kickoff") or items[0].get("date")) > now
        ]
        weeks = {int(r["week"]) for r in targets}
        if len(weeks) == 1:
            try:
                fresh = ESPNMarketClient(timeout_seconds=8).current_markets(
                    pl.DataFrame(targets), week=next(iter(weeks))
                )
                for q in fresh:
                    if q.market_type != "moneyline":
                        continue
                    sides = {
                        q.first_side: q.first_american_odds,
                        q.second_side: q.second_american_odds,
                    }
                    quotes[q.game_id].append(
                        {
                            "home_ml": sides.get("home"),
                            "away_ml": sides.get("away"),
                            "provider": q.book,
                            "captured_at": q.captured_at.isoformat(),
                            "source": q.provider,
                            "source_event_id": q.source_event_id,
                        }
                    )
            except (ValueError, OSError) as exc:
                source_errors.append(str(exc))
            now = datetime.now(UTC)
        elif targets:
            source_errors.append("Multiple weeks: cannot fetch a coherent current slate")
    counts = Counter()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for game, items in sorted(by_game.items()):
        row = items[0]
        p = probability(row.get("calibrated_home_probability"))
        kickoff = timestamp(row.get("kickoff") or row.get("date"))
        if kickoff >= EVALUATION_END:
            counts["outside_frozen_evaluation_window"] += 1
            continue
        if kickoff <= now:
            counts["already_started"] += 1
            continue
        if p is None or any(probability(x.get("calibrated_home_probability")) != p for x in items):
            counts["missing_or_conflicting_model"] += 1
            continue
        if sport == "cfb":
            quotes[game] = json.loads(row.get("market_quotes_json") or "[]")
        eligible = {}
        for q in quotes[game]:
            home, away = decimal(q.get("home_ml")), decimal(q.get("away_ml"))
            book = str(q.get("provider") or "").strip()
            # Anonymous provider IDs cannot establish independent named-book evidence.
            if home is None or away is None or not book or "book " in book.casefold():
                continue
            try:
                observed = timestamp(q.get("captured_at"))
            except (TypeError, ValueError):
                continue
            if not generated - timedelta(hours=6) <= observed <= now < kickoff:
                continue
            if now - observed > timedelta(minutes=120):
                continue
            if book not in eligible or observed > eligible[book][0]:
                eligible[book] = (observed, q, (1 / home) / (1 / home + 1 / away))
        if not eligible:
            counts["missing_named_paired_market"] += 1
            continue
        # Book selection is fixed by name, never outcome, price advantage or model edge.
        book = min(eligible, key=lambda b: (0 if "draft" in b.casefold() else 1, b.casefold()))
        observed, quote, market = eligible[book]
        identity = f"{SPEC}:{sport}:{game}"
        rid = hashlib.sha256(identity.encode()).hexdigest()[:24]
        record = {
            "spec": SPEC,
            "id": rid,
            "sport": sport,
            "game_id": game,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "kickoff": kickoff.isoformat(),
            "season": int(row["season"]),
            "week": int(row["week"]),
            "captured_at": now.isoformat(),
            "source_prediction_generated_at": generated.isoformat(),
            "prediction_source_sha256": source_hash,
            "experiment_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_version": audit.get("identity", {}).get("model_version")
            or (
                "asset-sha256:"
                + hashlib.sha256(Path("reports/current_model.json").read_bytes()).hexdigest()
                if Path("reports/current_model.json").exists()
                else "UNVERSIONED"
            ),
            "reference_book": book,
            "quote_collector_observed_at": observed.isoformat(),
            "original_paired_quote": quote,
            "source_quote_origin_verified": False,
            "source_quote_at": None,
            "predictions": forecasts(market, p),
            "stake_units": 0,
            "decision": "RESEARCH_OBSERVATION",
            "betting_authorized": False,
        }
        counts[
            "new_frozen"
            if immutable(root / "forecasts" / (rid + ".json"), record)
            else "already_frozen"
        ] += 1
    return {**dict(counts), "source_errors": source_errors}


def settle(root, now):
    cache, errors = {}, {}
    for path in sorted((root / "forecasts").glob("*.json")):
        row = json.loads(path.read_text())
        target = root / "grades" / path.name
        if target.exists() or timestamp(row["kickoff"]) >= now:
            continue
        commits = (
            subprocess.run(
                ["git", "log", "--format=%H %cI", "--", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
        if len(commits) != 1:
            errors[row["id"]] = "Missing unique publication commit"
            continue
        sha, stamp = commits[0].split(" ", 1)
        if not timestamp(row["captured_at"]) <= timestamp(stamp) < timestamp(row["kickoff"]):
            errors[row["id"]] = "Publication not before kickoff"
            continue
        key = (
            row["sport"],
            timestamp(row["kickoff"]).astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d"),
        )
        if key not in cache:
            if len(cache) >= 10:
                errors[row["id"]] = "Daily free request cap reached"
                continue
            try:
                cache[key] = fetch(*key)
            except (OSError, ValueError, TimeoutError) as exc:
                cache[key] = None
                errors[str(key)] = type(exc).__name__
        if cache[key] is None:
            continue
        payload, url = cache[key]
        final = final_result(row, payload)
        if final is None:
            continue
        y = (
            None
            if final["home_score"] == final["away_score"]
            else int(final["home_score"] > final["away_score"])
        )
        immutable(
            target,
            {
                "id": row["id"],
                "outcome": y,
                "final_score": final,
                "result_source": url,
                "graded_at": now.isoformat(),
                "pregame_publication_commit": sha,
                "forecast_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sport", choices=("nfl", "cfb"), required=True)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--refresh-market", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("history/market_first_v1"))
    parser.add_argument("--out", type=Path, default=Path("docs/market_first_experiment.json"))
    args = parser.parse_args()
    if args.historical:
        rows = nfl_history(args.historical)
        report = {
            "status": "EXPLORATORY_ARCHIVE_DIAGNOSTIC",
            "data_sha256": hashlib.sha256(args.historical.read_bytes()).hexdigest(),
            "historical_untouched_holdout": False,
            "all": scores(rows),
            "by_season": {
                str(s): scores([r for r in rows if r["season"] == s])
                for s in sorted({r["season"] for r in rows})
            },
        }
    else:
        now = datetime.now(UTC)
        coverage = capture(args.sport, args.root, now, refresh_market=args.refresh_market)
        now = datetime.now(UTC)
        errors = settle(args.root, now)
        rows = []
        for path in (args.root / "forecasts").glob("*.json"):
            grade_path = args.root / "grades" / path.name
            if grade_path.exists():
                row = json.loads(path.read_text())
                grade_record = json.loads(grade_path.read_text())
                if (
                    grade_record.get("forecast_sha256")
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    errors[row["id"]] = "Forecast changed after grading"
                    continue
                row.update(grade_record)
                rows.append(row)
        report = {
            "status": "PROSPECTIVE_RESEARCH_ONLY",
            "observed_at": now.isoformat(),
            "capture": coverage,
            "frozen_games": len(list((args.root / "forecasts").glob("*.json"))),
            "grading_errors": errors,
            "scores": scores(rows),
            "by_model_version": {
                v: scores([r for r in rows if r["model_version"] == v])
                for v in sorted({r["model_version"] for r in rows})
            },
            "by_season": {
                str(y): scores([r for r in rows if r["season"] == y])
                for y in sorted({r["season"] for r in rows})
            },
            "ties_omitted": sum(r.get("outcome") is None for r in rows),
        }
    report.update(
        {
            "spec": SPEC,
            "evaluation_end_exclusive": EVALUATION_END.isoformat(),
            "inference_status": "DESCRIPTIVE_UNTIL_FIXED_HORIZON_THEN_EXTERNAL_RELEASE_REVIEW",
            "sport": args.sport,
            "betting_authorized": False,
            "promotion_eligible": False,
            "stake_units": 0,
            "hypothesis": "Fixed 1.1 market log-odds may improve favorite/longshot calibration",
            "limitations": [
                "Advertised research prices are not verified executable fills",
                "Quote-origin timestamps missing; economic EV, ROI and CLV are not certified",
                "Historical archives already inspected; no pristine holdout claim",
                "No threshold/weight fitting; six paired tests use Bonferroni week-block intervals",
                "Prospective probability improvement alone cannot authorize bets",
            ],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "frozen_games": report.get("frozen_games"),
                "capture": report.get("capture"),
                "games": report.get("all", {}).get("games"),
            }
        )
    )


if __name__ == "__main__":
    main()
