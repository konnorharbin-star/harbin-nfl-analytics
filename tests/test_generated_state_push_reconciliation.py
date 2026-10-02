from __future__ import annotations

import os
import subprocess
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
ROOT = Path(__file__).resolve().parents[1]
PUSH_SCRIPT = ROOT / "scripts" / "push_generated_state.sh"

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
    (seed / "shared.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "shared.txt")
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

    (first / "first.txt").write_text("first\n", encoding="utf-8")
    _git(first, "add", "first.txt")
    _git(first, "commit", "-m", "first writer")

    (second / "second.txt").write_text("second\n", encoding="utf-8")
    _git(second, "add", "second.txt")
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
        ["git", "--git-dir", str(remote), "show", "main:first.txt"],
        cwd=tmp_path,
    ).stdout
    second_value = _run(
        ["git", "--git-dir", str(remote), "show", "main:second.txt"],
        cwd=tmp_path,
    ).stdout
    assert first_value == "first\n"
    assert second_value == "second\n"


def test_reconciler_fails_closed_on_true_content_conflict(tmp_path: Path) -> None:
    remote, _ = _seed_remote(tmp_path)
    first = _clone(remote, tmp_path / "first")
    second = _clone(remote, tmp_path / "second")

    (first / "shared.txt").write_text("first\n", encoding="utf-8")
    _git(first, "add", "shared.txt")
    _git(first, "commit", "-m", "first writer")

    (second / "shared.txt").write_text("second\n", encoding="utf-8")
    _git(second, "add", "shared.txt")
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
        ["git", "--git-dir", str(remote), "show", "main:shared.txt"],
        cwd=tmp_path,
    ).stdout
    assert remote_value == "second\n"
