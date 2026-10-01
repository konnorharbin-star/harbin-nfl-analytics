# Stage 9 — prospective recent-form totals shadow

Stage 8 found a small NFL-specific historical development signal for recent-form totals and froze it for 2026 shadow evaluation. Stage 9 turns that frozen candidate into a true prospective evidence stream without changing the canonical fair score or the market model.

## Frozen specification

The structure remains unchanged from Stage 8:

```text
recent_alpha = 0.20
features = [epa_per_play, success_rate]
ridge_alpha = 10.0
blend_weight = 0.50
training_seasons = 2022-2025
release_state = SHADOW
```

The specification is not retuned from 2026 results.

## Current-week attachment

`nfl/recent_form_current.py` rebuilds the selected recent-form signals using only completed regular-season game IDs before the target week. It refits the frozen residual structure on the fixed 2022-2025 development window and adds separate fields such as:

- `recent_form_total_adjustment`;
- `recent_form_shadow_total`;
- `recent_form_total_release_state`;
- frozen alpha/ridge/blend metadata.

The canonical `baseline_total` is never replaced. `nfl/market_intel.py` continues to use `baseline_total` for total probabilities, no-vig comparison, expected value, signal classification and staking. Recent-form fields are propagated only as diagnostics.

If the recent-form PBP/refit path fails, the canonical pipeline continues. The shadow fields become `BLOCKED`, the reason is added to source/data-quality metadata, and no recent-form prediction is eligible for prospective capture.

## Live current-slate audit

The Stage 9 pull-request live-source audit ran against the next unplayed 2026 regular-season slate and returned:

- season: `2026`;
- week: `4`;
- canonical games: `16`;
- recent-form covered games: `16`;
- recent-form coverage: `100%`;
- frozen training games: `831`;
- frozen training seasons: `2022-2025`;
- minimum shadow total: `41.2311`;
- maximum shadow total: `52.2904`;
- `baseline_total_unchanged`: `true`.

Those shadow totals are not betting recommendations and do not alter the canonical market path.

## Prospective first-snapshot ledger

`nfl/recent_form_forward.py` maintains:

```text
history/recent_form_shadow_predictions_v1.csv
```

A row can be appended only when:

1. the frozen shadow state is `SHADOW`;
2. canonical and shadow totals are finite;
3. the schedule has an explicit kickoff timestamp; and
4. capture time is strictly before kickoff.

The ledger keeps the first eligible prediction for each game/frozen specification. Repeated model runs do not overwrite or replace the first pregame snapshot. Post-kickoff captures are rejected.

The pull-request live audit is intentionally read-only and does not populate this ledger. The first canonical operational run on `main` after Stage 9 is merged is the beginning of prospective evidence capture.

## Independent grading

`grade_recent_form_forward.py` and the scheduled live-grading workflow grade only persisted pre-kickoff ledger rows after final scores are available.

The grader:

- excludes malformed/post-kickoff rows;
- keeps the first persisted snapshot for each game;
- verifies the row matches the frozen Stage 8 specification;
- joins only completed final scores;
- compares canonical total error with frozen shadow total error;
- uses the paired-bootstrap evidence gate from `nfl/shadow_gate.py`.

The promotion-evidence minimum remains `128` prospectively graded games. Even after that sample threshold is reached, promotion evidence requires:

- positive MAE point improvement;
- positive RMSE point improvement;
- the lower 95% confidence bound for MAE improvement above zero; and
- the lower 95% confidence bound for RMSE improvement above zero.

Anything weaker remains `SHADOW_FAILING` or `SHADOW_INCONCLUSIVE`.

## Evidence separation

Three different evidence classes remain distinct:

1. **2022-2025 rolling development evidence** — used to select/freeze the recent-form total structure;
2. **2026 reconstructed out-of-time shadow check** — useful monitoring context, but not a prospective ledger;
3. **2026 prospective first-snapshot evidence** — predictions actually persisted before kickoff after Stage 9 deployment.

Only the third class is eligible to accumulate true forward promotion evidence for this candidate.

## Current decision

- recent-form margin: **disabled**;
- recent-form total: **SHADOW**;
- canonical total: **unchanged**;
- canonical market probabilities/EV: **unchanged**;
- prospective grading: **enabled after deployment**;
- promotion: **not eligible**.
