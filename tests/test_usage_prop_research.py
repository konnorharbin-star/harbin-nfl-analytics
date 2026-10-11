"""Self-contained deterministic guards for usage challenger."""
from scripts.usage_prop_research import _usage_forecast, run_study


def row(week, market, targets=6.0, receptions=4.0, yards=48.0,
        carries=8.0, rush=32.0, attempts=30.0, passing=210.0):
    return {"game_id":f"g{week}","season":2024,"week":week,"team":"A",
            "player_id":"P","position":"WR","market":market,
            "targets":targets,"receptions":receptions,"receiving_yards":yards,
            "carries":carries,"rushing_yards":rush,"attempts":attempts,
            "passing_yards":passing}


def test_only_preweek_history_used():
    previous=[row(w,"receiving_yards") for w in range(1,5)]
    selected=row(5,"receiving_yards")
    base=_usage_forecast(selected,previous,3)
    previous.append(row(5,"receiving_yards",targets=1000,
                        receptions=1000,yards=100000))
    assert _usage_forecast(selected,previous,3)==base


def test_zero_opportunity_and_safe_smoothing():
    previous=[row(w,"receptions",targets=0,receptions=0,yards=0)
              for w in range(1,5)]
    assert _usage_forecast(row(5,"receptions"),previous,3)==0
    assert _usage_forecast(row(5,"receiving_yards"),previous,3)==0


def test_volume_and_conditional_yards():
    previous=[row(w,"receiving_yards") for w in range(1,5)]
    mean=_usage_forecast(row(5,"receiving_yards"),previous,3)
    assert 0<mean<100
    assert _usage_forecast(row(5,"receptions"),previous,3)>0


def test_missing_opportunities_never_imputed():
    previous=[row(w,"rushing_yards") for w in range(1,5)]
    for x in previous:
        x["carries"]=None
    assert _usage_forecast(row(5,"rushing_yards"),previous,3) is None


def test_staleness_blocks():
    previous=[row(w,"passing_yards") for w in range(1,5)]
    assert _usage_forecast(row(8,"passing_yards"),previous,3) is None


def test_unknown_or_bad_windows_block():
    prior=[row(w,"passing_yards") for w in range(1,5)]
    for bad in (2,6,9):
        try:
            _usage_forecast(row(5,"passing_yards"),prior,bad)
            assert False
        except ValueError:
            pass
