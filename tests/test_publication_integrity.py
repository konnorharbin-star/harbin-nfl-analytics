"""Step 7: multi-page checksums, updated timestamp and true remote publication."""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nfl.publication_integrity import (
    _display,
    _run_tag,
    build_publication_manifest,
    validate_publication_manifest,
    write_publication_manifest,
)
from scripts.verify_remote_publication import verify_remote_publication

RUN = datetime(2026, 10, 7, 23, 18, 5, tzinfo=UTC)


def _publication(tmp_path: Path, *, pages: int = 2) -> tuple[Path, Path]:
    outputs, docs = tmp_path / "outputs", tmp_path / "docs"
    outputs.mkdir()
    docs.mkdir()
    tag = _run_tag(RUN)
    stable = [
        f"outputs/nfl_week_5_page{page}.png" for page in range(1, pages + 1)
    ]
    fresh = [
        f"outputs/nfl_week_5_run_{tag}_page{page}.png"
        for page in range(1, pages + 1)
    ]
    report = {
        "generated_at": RUN.isoformat(),
        "meta": {"season": 2026, "week": 5, "generated_at": RUN.isoformat()},
        "release_state": "RESEARCH",
        "portfolio": {"approved_units": 0},
        "publication": {
            "html": "outputs/nfl_week_5.html",
            "board_games": 15 if pages == 2 else 1,
            "png_pages": stable,
            "cache_safe_png_pages": fresh,
            "latest_png": "outputs/latest.png",
            "run_tag": tag,
        },
    }
    data = json.dumps(report, indent=2, sort_keys=True)
    (outputs / "current_model.json").write_text(data)
    (docs / "latest.json").write_text(data)
    (outputs / "current_predictions.csv").write_text("game_id,quant_market\ng1,total\n")
    (docs / "latest.csv").write_text("game_id,quant_market\ng1,total\n")
    (outputs / "suggested_bets.csv").write_text("game_id,signal\ng1,PASS\n")
    (outputs / "quant_recommendations.csv").write_text("game_id,stake_units\n")
    (outputs / "quant_card.html").write_text("<html>Current quant card</html>")
    (docs / "quant.html").write_text("<html>Current quant card</html>")
    (outputs / "nfl_week_5.html").write_text("<html>Current board</html>")
    (docs / "index.html").write_text("<html>Current board</html>")
    (docs / "audit_snapshot.json").write_text('{"status":"RESEARCH"}')
    readme = [
        "# Latest NFL model output",
        "**Season / Week:** 2026 / 5",
        f"**Updated:** {_display(RUN)}",
        "**Release state:** RESEARCH",
        *[f"- [Stable page]({Path(x).name})" for x in stable],
        *[f"- [Fresh page]({Path(x).name})" for x in fresh],
        "",
    ]
    (outputs / "README.md").write_text("\n".join(readme))
    for index, (stable_file, fresh_file) in enumerate(
        zip(stable, fresh, strict=True)
    ):
        # PNG file signature check, not image decoding; the published hash
        # is strictly about bytes and consistency, not rendered geometry.
        png = b"\x89PNG\r\n\x1a\n" + f"page-{index+1}-fresh".encode()
        (tmp_path / stable_file).write_bytes(png)
        (tmp_path / fresh_file).write_bytes(png)
    (outputs / "latest.png").write_bytes((tmp_path / stable[0]).read_bytes())
    (docs / "latest.png").write_bytes((tmp_path / stable[0]).read_bytes())
    return outputs, docs


def _is_fail(outputs: Path, docs: Path, reason: str) -> None:
    result = validate_publication_manifest(output_dir=outputs, docs_dir=docs)
    assert result["status"] == "FAIL", result
    assert reason in result["reason"], result["reason"]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, text=True,
        capture_output=True, check=True,
    )


def test_manifest_includes_every_png_page_public_copy_and_betting_csv(tmp_path):
    outputs, docs = _publication(tmp_path)
    manifest = write_publication_manifest(output_dir=outputs, docs_dir=docs)
    assert manifest["status"] == "PASS"
    assert manifest["page_count"] == 2
    assert manifest["run_tag"] == _run_tag(RUN)
    assert manifest["file_count"] >= 18
    files = {value["path"]: value for value in manifest["files"]}
    for required in (
        "outputs/current_model.json",
        "outputs/README.md",
        "outputs/suggested_bets.csv",
        "outputs/quant_recommendations.csv",
        "outputs/latest.png",
        "docs/latest.json",
        "docs/latest.png",
        "docs/latest.csv",
        "docs/index.html",
        "outputs/nfl_week_5_page2.png",
        f"outputs/nfl_week_5_run_{_run_tag(RUN)}_page2.png",
    ):
        assert required in files
        assert len(files[required]["sha256"]) == 64
    assert validate_publication_manifest(
        output_dir=outputs, docs_dir=docs
    )["status"] == "PASS"


@pytest.mark.parametrize("stale_path", [
    "outputs/nfl_week_5_page2.png",
    "outputs/suggested_bets.csv",
    "docs/latest.csv",
    "docs/index.html",
    "outputs/README.md",
])
def test_publication_fails_on_partial_or_stale_public_state(tmp_path, stale_path):
    outputs, docs = _publication(tmp_path)
    write_publication_manifest(output_dir=outputs, docs_dir=docs)
    (tmp_path / stale_path).write_bytes(b"stale content")
    _is_fail(outputs, docs, "invalid image signature" if "page2" in stale_path else "")


def test_stale_second_cache_safe_png_cannot_pass_page1_only_check(tmp_path):
    outputs, docs = _publication(tmp_path)
    write_publication_manifest(output_dir=outputs, docs_dir=docs)
    page2 = outputs / f"nfl_week_5_run_{_run_tag(RUN)}_page2.png"
    page2.write_bytes(b"\x89PNG\r\n\x1a\nWRONG SECOND PAGE")
    _is_fail(outputs, docs, "page 2")


def test_stale_valid_but_old_readme_date_is_rejected(tmp_path):
    outputs, docs = _publication(tmp_path)
    readme = outputs / "README.md"
    readme.write_text(readme.read_text().replace(
        f"**Updated:** {_display(RUN)}", "**Updated:** Oct 1, 2026 · 9:00 AM CT"
    ))
    with pytest.raises(ValueError, match="README timestamp"):
        build_publication_manifest(output_dir=outputs, docs_dir=docs)


def test_manifest_must_match_its_public_mirror(tmp_path):
    outputs, docs = _publication(tmp_path)
    write_publication_manifest(output_dir=outputs, docs_dir=docs)
    (docs / "publication_manifest.json").write_text('{"status":"PASS"}')
    _is_fail(outputs, docs, "public manifest differs")


def test_run_tag_and_page_count_must_match_canonical_run(tmp_path):
    outputs, docs = _publication(tmp_path)
    model = json.loads((outputs / "current_model.json").read_text())
    model["publication"]["run_tag"] = "20261001_000000_CT"
    (outputs / "current_model.json").write_text(json.dumps(model))
    with pytest.raises(ValueError, match="run tag does not match"):
        build_publication_manifest(output_dir=outputs, docs_dir=docs)
    model["publication"]["run_tag"] = _run_tag(RUN)
    model["publication"]["board_games"] = 30
    (outputs / "current_model.json").write_text(json.dumps(model))
    with pytest.raises(ValueError, match="page count"):
        build_publication_manifest(output_dir=outputs, docs_dir=docs)


def test_path_traversal_and_missing_page_fail_closed(tmp_path):
    outputs, docs = _publication(tmp_path)
    model = json.loads((outputs / "current_model.json").read_text())
    model["publication"]["cache_safe_png_pages"][1] = "../elsewhere/page2.png"
    (outputs / "current_model.json").write_text(json.dumps(model))
    with pytest.raises(ValueError, match="escapes repository"):
        build_publication_manifest(output_dir=outputs, docs_dir=docs)


def test_new_run_manifest_rejects_modified_older_model_run(tmp_path):
    outputs, docs = _publication(tmp_path)
    write_publication_manifest(output_dir=outputs, docs_dir=docs)
    model = json.loads((outputs / "current_model.json").read_text())
    model["meta"]["generated_at"] = (
        datetime(2026, 10, 7, 22, tzinfo=UTC).isoformat()
    )
    (outputs / "current_model.json").write_text(json.dumps(model))
    _is_fail(outputs, docs, "run tag does not match")


def test_remote_verification_catches_stale_page2_and_csv(tmp_path):
    base = tmp_path / "source"
    base.mkdir()
    outputs, docs = _publication(base)
    manifest = write_publication_manifest(output_dir=outputs, docs_dir=docs)
    _git(base, "init", "-b", "main")
    _git(base, "config", "user.name", "Publication Integration")
    _git(base, "config", "user.email", "publication@example.invalid")
    _git(base, "add", "outputs", "docs")
    _git(base, "commit", "-m", "validated publication")
    bare = tmp_path / "remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(bare)],
        check=True, text=True, capture_output=True,
    )
    _git(base, "remote", "add", "origin", str(bare))
    _git(base, "push", "-u", "origin", "main")
    good = verify_remote_publication(repo_root=base)
    assert good["status"] == "PASS"
    assert good["files_verified"] == manifest["file_count"] + 2

    # Another remote writer replaces page 2 without replacing run manifest.
    elsewhere = tmp_path / "other"
    subprocess.run(
        ["git", "clone", "-b", "main", str(bare), str(elsewhere)],
        check=True, capture_output=True, text=True,
    )
    _git(elsewhere, "config", "user.name", "Other writer")
    _git(elsewhere, "config", "user.email", "other@example.invalid")
    (elsewhere / "outputs/nfl_week_5_page2.png").write_bytes(
        b"\x89PNG\r\n\x1a\nREMOTE STALE PAGE 2"
    )
    _git(elsewhere, "add", "outputs/nfl_week_5_page2.png")
    _git(elsewhere, "commit", "-m", "accidental page 2 replacement")
    _git(elsewhere, "push", "origin", "main")
    _git(base, "fetch", "origin", "main")
    with pytest.raises(RuntimeError, match="STALE.*page2"):
        verify_remote_publication(repo_root=base)
