#!/usr/bin/env bash
set -euo pipefail

remote="${1:-origin}"
target_branch="${2:-${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-main}}}"
max_attempts="${3:-8}"

if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::max_attempts must be a positive integer"
  exit 2
fi

if git diff --quiet && git diff --cached --quiet; then
  :
else
  echo "::error::push_generated_state.sh expects committed changes only"
  exit 2
fi

source_base="$(git rev-parse HEAD^)"
is_generated_path() {
  case "$1" in
    outputs/*|history/*|reports/*)
      return 0
      ;;
    docs/*.json|docs/*.html|docs/*.csv|docs/*.png)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

for attempt in $(seq 1 "$max_attempts"); do
  echo "Generated-state push attempt ${attempt}/${max_attempts} -> ${remote}/${target_branch}"

  git fetch "$remote" "$target_branch"

  if ! git merge-base --is-ancestor "$source_base" FETCH_HEAD; then
    echo "::error::Generated-state publication is stale because the target branch no longer descends from the writer's source revision."
    exit 3
  fi

  unsafe_changes=""
  while IFS= read -r path; do
    [ -z "$path" ] && continue
    if ! is_generated_path "$path"; then
      unsafe_changes="${unsafe_changes}${unsafe_changes:+, }${path}"
    fi
  done < <(git diff --name-only "$source_base" FETCH_HEAD)

  if [ -n "$unsafe_changes" ]; then
    echo "::error::Generated-state publication is stale because newer source changes supersede this run: $unsafe_changes"
    exit 3
  fi

  if ! git rebase FETCH_HEAD; then
    git rebase --abort || true
    echo "::error::Generated-state rebase conflict against ${remote}/${target_branch}; refusing to overwrite another writer."
    exit 1
  fi

  if git push "$remote" "HEAD:$target_branch"; then
    echo "Generated-state push succeeded."
    exit 0
  fi

  if [ "$attempt" -lt "$max_attempts" ]; then
    sleep_seconds=$((attempt * 2))
    echo "Push raced another writer; retrying after ${sleep_seconds}s."
    sleep "$sleep_seconds"
  fi
done

echo "::error::Generated-state push failed after ${max_attempts} attempts."
exit 1
