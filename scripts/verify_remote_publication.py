"""Verify the COMPLETE NFL model publication reached the target Git branch.

Run after scripts/push_generated_state.sh; never interpret a successful local
render, checksum or commit as proof that public GitHub contains that run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from nfl.publication_integrity import validate_publication_manifest


def _git_blob(repo: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=repo,
        stderr=subprocess.PIPE,
    )


def verify_remote_publication(
    *,
    repo_root: str | Path = ".",
    branch: str = "main",
    remote: str = "origin",
) -> dict[str, object]:
    root = Path(repo_root).resolve()
    result = validate_publication_manifest(
        output_dir=root / "outputs", docs_dir=root / "docs"
    )
    if result["status"] != "PASS":
        raise RuntimeError("local publication integrity failure: " + str(result["reason"]))
    local_manifest_bytes = (root / "outputs/publication_manifest.json").read_bytes()
    manifest = json.loads(local_manifest_bytes)
    revision = f"{remote}/{branch}"
    paths = [
        "outputs/publication_manifest.json",
        "docs/publication_manifest.json",
        *(item["path"] for item in manifest["files"]),
    ]
    for path in paths:
        source = root / path
        try:
            remote_bytes = _git_blob(root, revision, path)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"published NFL file is missing at {revision}:{path}"
            ) from exc
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(
            remote_bytes
        ).digest():
            raise RuntimeError(
                f"published NFL file is STALE at {revision}:{path}"
            )
    return {
        "status": "PASS",
        "remote_revision": revision,
        "files_verified": len(paths),
        "model_generated_at": manifest["generated_at"],
        "run_tag": manifest["run_tag"],
        "page_count": manifest["page_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    print(json.dumps(verify_remote_publication(
        repo_root=args.repo_root,
        branch=args.branch,
        remote=args.remote,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
