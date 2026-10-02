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

for attempt in $(seq 1 "$max_attempts"); do
  echo "Generated-state push attempt ${attempt}/${max_attempts} -> ${remote}/${target_branch}"

  git fetch "$remote" "$target_branch"

  if ! git rebase "$remote/$target_branch"; then
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
