from __future__ import annotations

import polars as pl

from nfl.recent_form_shadow import (
    FROZEN_BLEND_WEIGHT,
    FROZEN_FEATURE_SET,
    FROZEN_RECENT_ALPHA,
    FROZEN_RIDGE_ALPHA,
    apply_frozen_recent_total,
    fit_frozen_recent_total,
)


def _training() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(80):
        epa = ((index % 11) - 5) / 10.0
        success = ((index % 7) - 3) / 10.0
        residual = (3.0 * epa) + (1.5 * success)
        rows.append(
            {
                "recent_epa_per_play_total_signal": epa,
                "recent_success_rate_total_signal": success,
                "total_residual": residual,
                "baseline_total": 44.0,
            }
        )
    return pl.DataFrame(rows)


def test_frozen_recent_form_structure_matches_development_selection() -> None:
    assert FROZEN_RECENT_ALPHA == 0.20
    assert FROZEN_FEATURE_SET == ("epa_per_play", "success_rate")
    assert FROZEN_RIDGE_ALPHA == 10.0
    assert FROZEN_BLEND_WEIGHT == 0.50


def test_shadow_application_preserves_canonical_total() -> None:
    training = _training()
    model = fit_frozen_recent_total(training)
    target = pl.DataFrame(
        {
            "recent_epa_per_play_total_signal": [0.4, -0.3],
            "recent_success_rate_total_signal": [0.2, -0.1],
            "baseline_total": [45.0, 41.0],
        }
    )
    scored = apply_frozen_recent_total(model, target)

    assert scored.get_column("baseline_total").to_list() == [45.0, 41.0]
    assert "recent_form_total_adjustment" in scored.columns
    assert "recent_form_shadow_total" in scored.columns
    assert scored.get_column("recent_form_total_release_state").to_list() == [
        "SHADOW",
        "SHADOW",
    ]
    assert scored.get_column("recent_form_shadow_total").to_list() != [45.0, 41.0]
