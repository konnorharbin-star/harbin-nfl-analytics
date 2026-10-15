from copy import deepcopy

import pytest

from nfl.player_workload_shrinkage import history_index, prior_mean, reweight, upgrade


def fixture():
    roles = [
        dict(team="A", player_id=p, role="RB", target_share=t, carry_share=c, qb_pass_share=0.0)
        for p, t, c in [("p1", 0.8, 0.2), ("p2", 0.2, 0.8)]
    ]
    d = {
        f"A:{p}:{m}": {"mean": v}
        for p, values in [("p1", [8.0, 2.0]), ("p2", [2.0, 8.0])]
        for m, v in zip(("receptions", "carries"), values, strict=True)
    }
    rows = [
        dict(
            season_type="REG",
            season=2026,
            week=w,
            team="A",
            player_id=p,
            receptions=rec,
            carries=car,
        )
        for w in (1, 2, 3, 4)
        for p, rec, car in [("p1", 2.0, 8.0), ("p2", 8.0, 2.0)]
    ]
    settings = {f: dict(window=8, blend=0.5) for f in ("receiving", "rushing")}
    return roles, d, rows, settings


def test_shrinks_routing_and_preserves_team_workload_and_qb_fields():
    roles, d, rows, settings = fixture()
    original = deepcopy(roles)
    revised, audit = reweight(roles, d, history_index(rows), season=2026, week=5, settings=settings)
    assert roles == original
    assert [r["target_share"] for r in revised] == [0.5, 0.5]
    assert [r["carry_share"] for r in revised] == [0.5, 0.5]
    assert all(r["qb_pass_share"] == 0 for r in revised)
    assert all(not r["fallback"] for r in audit)


def test_future_and_other_team_records_cannot_change_weights():
    roles, d, rows, settings = fixture()
    before, _ = reweight(roles, d, history_index(rows), season=2026, week=5, settings=settings)
    contaminated = rows + [
        dict(rows[0], week=5, receptions=999, carries=999),
        dict(rows[1], week=6, receptions=999, carries=999),
        dict(rows[0], team="B", receptions=999, carries=999),
    ]
    after, _ = reweight(
        roles, d, history_index(contaminated), season=2026, week=5, settings=settings
    )
    assert before == after


def test_missing_or_stale_history_falls_back_without_inventing_own_stats():
    roles, d, rows, settings = fixture()
    assert prior_mean(history_index(rows), 2026, 8, "A", "p1", "receptions", 8) is None
    revised, audit = reweight(roles, d, {}, season=2026, week=5, settings=settings)
    assert revised == roles
    assert all(a["prior_mean"] is None and a["fallback"] for a in audit)
    d["A:p1:receptions"]["mean"] = 7.9
    assert reweight(roles, d, {}, season=2026, week=5, settings=settings)[0] == roles


def test_zero_blend_is_exact_original_and_unallocated_mass_is_preserved():
    roles, d, rows, settings = fixture()
    for s in settings.values():
        s["blend"] = 0.0
    revised, _ = reweight(roles, d, history_index(rows), season=2026, week=5, settings=settings)
    assert revised == roles
    settings["receiving"]["blend"] = 0.5
    d["A:OTHER:receptions"] = {"mean": 2.0}
    revised, _ = reweight(roles, d, history_index(rows), season=2026, week=5, settings=settings)
    assert sum(r["target_share"] for r in revised) < 1.0


def test_duplicate_source_player_week_is_rejected():
    _, _, rows, _ = fixture()
    with pytest.raises(ValueError, match="DUPLICATE_PLAYER_WEEK"):
        history_index(rows + [rows[0]])


def test_bad_settings_or_missing_baseline_are_rejected():
    roles, d, rows, settings = fixture()
    settings["receiving"]["blend"] = 2.0
    with pytest.raises(ValueError, match="INVALID_FROZEN"):
        reweight(roles, d, history_index(rows), season=2026, week=5, settings=settings)
    settings["receiving"]["blend"] = 0.5
    with pytest.raises(ValueError, match="MISSING_BASELINE"):
        reweight(roles, {}, history_index(rows), season=2026, week=5, settings=settings)


def test_invalid_parent_cannot_become_new_version():
    with pytest.raises(ValueError, match="INVALID_PARENT"):
        upgrade({}, {}, {}, at=None, settings={}, model_sha="x", source_sha="y")
