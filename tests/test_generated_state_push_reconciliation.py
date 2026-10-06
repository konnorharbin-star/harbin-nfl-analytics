from __future__ import annotations

import os
import subprocess
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
ROOT = Path(__file__).resolve().parents[1]
PUSH_SCRIPT = ROOT / "scripts" / "push_generated_state.sh"

EXPECTED_WRITERS = {
    "candidate-benchmark.yml",
    "espn-verified-market-backtest.yml",
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
    "regime-edge-reliability.yml",
    "research-status.yml",
    "verified-market-backtest.yml",
}


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
        env=env,
    )


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, check=check)


def _configure_repo(path: Path) -> None:
    _git(path, "config", "user.name", "Stage36 Test")
    _git(path, "config", "user.email", "stage36@example.invalid")


def _seed_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    _run(["git", "init", "--bare", str(remote)], cwd=tmp_path)
    _run(["git", "init", "-b", "main", str(seed)], cwd=tmp_path)
    _configure_repo(seed)
    (seed / "nfl").mkdir()
    (seed / "reports").mkdir()
    (seed / "nfl" / "source.py").write_text("VALUE = 'base'\n", encoding="utf-8")
    (seed / "reports" / "shared.json").write_text('{"value":"base"}\n', encoding="utf-8")
    _git(seed, "add", "nfl/source.py", "reports/shared.json")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "-u", "origin", "main")
    return remote, seed


def _clone(remote: Path, path: Path) -> Path:
    _run(["git", "clone", "-b", "main", str(remote), str(path)], cwd=path.parent)
    _configure_repo(path)
    return path


def _writer_workflows() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if "git push" in text or "push_generated_state.sh" in text:
            output[path.name] = text
    return output


def test_all_generated_state_writers_use_reconciler() -> None:
    writers = _writer_workflows()
    assert set(writers) == EXPECTED_WRITERS
    for name, text in writers.items():
        assert "bash scripts/push_generated_state.sh" in text, name
        assert "nfl-generated-state-" not in text, name
        assert "cancel-in-progress: true" in text, name
        assert "github.event.workflow_run.head_branch" in text, name


def test_reconciler_rebases_non_conflicting_cross_writer_race(tmp_path: Path) -> None:
    remote, _ = _seed_remote(tmp_path)
    first = _clone(remote, tmp_path / "first")
    second = _clone(remote, tmp_path / "second")

    (first / "reports" / "first.json").write_text(
        '{"value":"first"}\n',
        encoding="utf-8",
    )
    _git(first, "add", "reports/first.json")
    _git(first, "commit", "-m", "first writer")

    (second / "outputs").mkdir()
    (second / "outputs" / "second.json").write_text(
        '{"value":"second"}\n',
        encoding="utf-8",
    )
    _git(second, "add", "outputs/second.json")
    _git(second, "commit", "-m", "second writer")
    _git(second, "push", "origin", "main")

    env = os.environ.copy()
    env["GITHUB_REF_NAME"] = "main"
    result = _run(
        ["bash", str(PUSH_SCRIPT), "origin", "main", "3"],
        cwd=first,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    first_value = _run(
        ["git", "--git-dir", str(remote), "show", "main:reports/first.json"],
        cwd=tmp_path,
    ).stdout
    second_value = _run(
        ["git", "--git-dir", str(remote), "show", "main:outputs/second.json"],
        cwd=tmp_path,
    ).stdout
    assert first_value == '{"value":"first"}\n'
    assert second_value == '{"value":"second"}\n'


def test_reconciler_fails_closed_on_true_content_conflict(tmp_path: Path) -> None:
    remote, _ = _seed_remote(tmp_path)
    first = _clone(remote, tmp_path / "first")
    second = _clone(remote, tmp_path / "second")

    (first / "reports" / "shared.json").write_text(
        '{"value":"first"}\n',
        encoding="utf-8",
    )
    _git(first, "add", "reports/shared.json")
    _git(first, "commit", "-m", "first writer")

    (second / "reports" / "shared.json").write_text(
        '{"value":"second"}\n',
        encoding="utf-8",
    )
    _git(second, "add", "reports/shared.json")
    _git(second, "commit", "-m", "second writer")
    _git(second, "push", "origin", "main")

    env = os.environ.copy()
    env["GITHUB_REF_NAME"] = "main"
    result = _run(
        ["bash", str(PUSH_SCRIPT), "origin", "main", "1"],
        cwd=first,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    remote_value = _run(
        ["git", "--git-dir", str(remote), "show", "main:reports/shared.json"],
        cwd=tmp_path,
    ).stdout
    assert remote_value == '{"value":"second"}\n'


def test_reconciler_skips_stale_output_after_source_change(tmp_path: Path) -> None:
    remote, _ = _seed_remote(tmp_path)
    stale = _clone(remote, tmp_path / "stale")
    source = _clone(remote, tmp_path / "source")

    (stale / "reports" / "stale.json").write_text(
        '{"value":"stale"}\n',
        encoding="utf-8",
    )
    _git(stale, "add", "reports/stale.json")
    _git(stale, "commit", "-m", "stale generated state")

    (source / "nfl" / "source.py").write_text(
        "VALUE = 'new-source'\n",
        encoding="utf-8",
    )
    _git(source, "add", "nfl/source.py")
    _git(source, "commit", "-m", "new source revision")
    _git(source, "push", "origin", "main")

    env = os.environ.copy()
    env["GITHUB_REF_NAME"] = "main"
    result = _run(
        ["bash", str(PUSH_SCRIPT), "origin", "main", "3"],
        cwd=stale,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "newer source changes supersede this run" in result.stdout
    missing = _run(
        [
            "git",
            "--git-dir",
            str(remote),
            "show",
            "main:reports/stale.json",
        ],
        cwd=tmp_path,
        check=False,
    )
    assert missing.returncode != 0
    source_value = _run(
        ["git", "--git-dir", str(remote), "show", "main:nfl/source.py"],
        cwd=tmp_path,
    ).stdout
    assert source_value == "VALUE = 'new-source'\n"


def test_espn_historical_writer_is_serialized_after_free_evidence() -> None:
    free = (WORKFLOW_DIR / "free-market-backtest.yml").read_text(
        encoding="utf-8"
    )
    espn = (WORKFLOW_DIR / "espn-verified-market-backtest.yml").read_text(
        encoding="utf-8"
    )
    model = (WORKFLOW_DIR / "nfl-model.yml").read_text(encoding="utf-8")

    assert 'workflows:\n      - "Free NFL Market Backtest"' in espn
    assert "github.event.workflow_run.conclusion == 'success'" in espn
    assert "github.event.workflow_run.event != 'schedule'" in espn
    assert "ref: main" in espn
    assert '"nfl/espn_historical.py"' in free
    assert '"run_espn_verified_market_backtest.py"' in free
    assert '"Free ESPN Historical Market Backtest"' in model
