#!/usr/bin/env bash
# Pages discovery can lag a successfully finalized upload. Bind to this upload,
# rather than accepting a same-name artifact from a previous workflow attempt.
set -euo pipefail
: "${PAGES_ARTIFACT_ID:?Uploaded Pages artifact ID is required}"
: "${GITHUB_REPOSITORY:?Repository is required}"
: "${GITHUB_RUN_ID:?Workflow run is required}"
[[ "$PAGES_ARTIFACT_ID" =~ ^[0-9]+$ ]] || exit 1
for attempt in {1..20}; do
  if available=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/artifacts" \
      --jq ".artifacts[] | select(.id == $PAGES_ARTIFACT_ID and .expired == false) | .id"); then
    if [[ "$available" == "$PAGES_ARTIFACT_ID" ]]; then
      # The deploy action uses a separate artifact discovery endpoint.
      sleep 3
      exit 0
    fi
  fi
  sleep 3
done
printf 'The uploaded Pages artifact %s is not discoverable after 20 checks.\n' "$PAGES_ARTIFACT_ID" >&2
exit 1
