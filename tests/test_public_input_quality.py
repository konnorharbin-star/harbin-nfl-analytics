import hashlib
import json
from datetime import timedelta

import pytest

from nfl.public_context_inputs import collect_context_inputs, context_contract
from nfl.public_input_quality import publication_quality, quality_report
from scripts.nfl_publish_current_dashboard import publish
from tests.test_nfl_publish_current_dashboard import sample, seed_assets
from tests.test_public_context_inputs import AT, GAME, PLAYER, fixture


def bundle_and_quality():
    bundle = sample()
    bundle["player_props"] = [PLAYER]
    bundle["generated_utc"] = (AT - timedelta(minutes=2)).isoformat()
    bundle["model_capture_utc"] = (AT - timedelta(minutes=5)).isoformat()
    raw = json.dumps(bundle).encode()
    digest = hashlib.sha256(raw).hexdigest()
    observation, _, read = fixture()
    context = collect_context_inputs(GAME, observation, [PLAYER], read=read, now=lambda: AT)
    view = {
        **GAME,
        "structured_game_observation": observation,
        "verified_context_inputs": context,
        "input_data_contract": context_contract(observation, context, GAME, asof=AT),
    }
    report = quality_report(
        {"capture_utc": AT.isoformat(), "collection_started_utc": AT.isoformat(), "games": [view]},
        bundle,
        digest,
    )
    return bundle, raw, digest, report


def test_quality_is_supplemental_exact_forecast_bound_and_rechecked_at_publication():
    bundle, raw, digest, report = bundle_and_quality()
    before = json.dumps(bundle).encode()
    view = publication_quality(report, bundle, forecast_sha256=digest, at=AT)
    assert view["status"] == "SUPPLEMENTAL_INPUT_CHECKS_AVAILABLE"
    assert view["games"][0]["contract"]["reported_injury_statuses_verified"] == 1
    assert view["games"][0]["contract"]["outdoor_weather_verified"]
    assert json.dumps(bundle).encode() == before == raw
    later = publication_quality(report, bundle, forecast_sha256=digest, at=AT + timedelta(hours=2))
    assert later["games"][0]["contract"]["event_binding_passed"] is False
    assert later["games"][0]["contract"]["outdoor_weather_verified"] is False


@pytest.mark.parametrize(
    "change",
    [
        lambda q: q.update(source_forecast_sha256="f" * 64),
        lambda q: q.update(source_forecast_generated_utc="2026-10-19T10:00:00Z"),
        lambda q: q.update(captured_utc="2026-10-20T01:00:00Z"),
        lambda q: q.update(collection_started_utc="2026-10-19T09:00:00Z"),
        lambda q: q["games"][0].update(away_team="KC"),
        lambda q: q["games"][0].update(kickoff_utc="2026-10-21T00:15:00Z"),
        lambda q: q["games"].append(q["games"][0]),
        lambda q: q.update(games=[]),
        lambda q: q.update(forecast_values_changed=True),
    ],
)
def test_mismatched_future_incomplete_and_ambiguous_reports_are_unavailable(change):
    bundle, _, digest, report = bundle_and_quality()
    change(report)
    result = publication_quality(report, bundle, forecast_sha256=digest, at=AT)
    assert result["status"] == "INPUT_CHECKS_UNAVAILABLE"
    assert result["games"] == []


def test_stale_report_never_keeps_prior_pass_flags():
    bundle, _, digest, report = bundle_and_quality()
    bundle["generated_utc"] = report["source_forecast_generated_utc"] = (
        AT - timedelta(hours=9)
    ).isoformat()
    bundle["model_capture_utc"] = report["source_forecast_model_capture_utc"] = (
        AT - timedelta(hours=10)
    ).isoformat()
    report["captured_utc"] = report["collection_started_utc"] = (
        AT - timedelta(hours=7)
    ).isoformat()
    assert (
        publication_quality(report, bundle, forecast_sha256=digest, at=AT)["reason"]
        == "INPUT_REPORT_STALE"
    )


def test_source_capture_cannot_be_later_than_the_report_containing_it():
    bundle, _, digest, report = bundle_and_quality()
    report["games"][0]["verified_context_inputs"]["availability"]["source"]["captured_utc"] = (
        AT + timedelta(seconds=1)
    ).isoformat()
    result = publication_quality(
        report, bundle, forecast_sha256=digest, at=AT + timedelta(seconds=2)
    )
    assert result["status"] == "INPUT_CHECKS_UNAVAILABLE"


def test_publisher_displays_quality_and_preserves_original_forecast_bytes(tmp_path):
    bundle, raw, _, report = bundle_and_quality()
    source = seed_assets(tmp_path, bundle)
    (source / "independent_input_quality.json").write_text(json.dumps(report), encoding="utf-8")
    docs = tmp_path / "docs"
    manifest = publish(source, docs, at=AT)
    page = (docs / "index.html").read_text(encoding="utf-8")
    assert "Confirmed starting quarterbacks" in page
    assert "Final player availability" in page
    assert "Verified forecast" in page
    assert "reported_injury_statuses_verified" not in page
    assert "did not change its values" in page
    assert (docs / "independent_latest.json").read_bytes() == raw
    assert manifest["input_quality_status"] == "SUPPLEMENTAL_INPUT_CHECKS_AVAILABLE"
    evidence = json.loads((docs / "independent_input_quality.json").read_bytes())
    assert evidence["source_evidence"]["source_forecast_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("content", [None, "invalid JSON", '{"spec":"wrong"}'])
def test_bad_or_missing_quality_cannot_leave_an_old_success_on_dashboard(tmp_path, content):
    bundle, _, _, _ = bundle_and_quality()
    source = seed_assets(tmp_path, bundle)
    if content is not None:
        (source / "independent_input_quality.json").write_text(content, encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "independent_input_quality.json").write_text('{"status":"OLD_SUCCESS"}')
    manifest = publish(source, docs, at=AT)
    assert manifest["input_quality_status"] == "INPUT_CHECKS_UNAVAILABLE"
    current = json.loads((docs / "independent_input_quality.json").read_bytes())
    assert current["status"] == "INPUT_CHECKS_UNAVAILABLE" and current["games"] == []
    assert "source_evidence" not in current
    assert "Current input checks unavailable" in (docs / "index.html").read_text(encoding="utf-8")
