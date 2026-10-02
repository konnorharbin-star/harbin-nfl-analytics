from __future__ import annotations

from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
EXPECTED_WRITERS = {
    "candidate-benchmark.yml",
    "forward-ledger-health.yml",
    "forward-shadow-summary.yml",
    "free-market-backtest.yml",
    "line-capture.yml",
    "live-grade.yml",
    "market-edge-shrinkage.yml",
    "nfl-model.yml",
    "probability-forward-capture.yml",
    "probability-forward-grade.yml",
    "qb-total-forward-capture.yml",
    "qb-total-forward-grade.yml",
    "research-status.yml",
}


def _writer_workflows() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "git push" in text:
            output[path.name] = text
    return output


def test_all_generated_state_writers_are_registered() -> None:
    writers = _writer_workflows()
    assert set(writers) == EXPECTED_WRITERS


def test_main_generated_state_writers_share_one_serial_lane() -> None:
    for name, text in _writer_workflows().items():
        assert "format('nfl-generated-state-{0}', github.ref_name)" in text, name
        assert (
            "format('nfl-generated-state-{0}', github.event.workflow_run.head_branch)"
            in text
        ), name
        assert (
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}"
            in text
        ), name


def test_pull_request_writer_validation_remains_workflow_scoped() -> None:
    for name, text in _writer_workflows().items():
        assert "github.event_name == 'pull_request'" in text, name
        assert "-pr-{0}" in text, name
