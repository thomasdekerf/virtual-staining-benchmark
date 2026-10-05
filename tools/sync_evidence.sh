#!/usr/bin/env bash
set -euo pipefail
benchmark_root=$(cd "$(dirname "$0")/.." && pwd)
benchmark_host=${BENCHMARK_SSH_HOST:-sofia}
benchmark_remote=${BENCHMARK_REMOTE_ROOT:-/sofia/projects/2026_059/virtual_staining}
for cohort in orion finland skin; do
  case "$cohort" in
    orion) source_dir="$benchmark_remote/runs/" ;;
    finland) source_dir="$benchmark_remote/unstained_finland_20261002_v1/runs/" ;;
    skin) source_dir="$benchmark_remote/unstained_dermarepo_20261002_v1/runs/" ;;
  esac
  mkdir -p "$benchmark_root/.source-cache/$cohort"
  rsync -az --timeout=60 -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
    --exclude='*smoke*/' --exclude='*profile*/' --exclude='hematoxylin_segmenter/' --exclude='segmenter/' \
    --include='*/' --include='config.yml' --include='status.json' --include='history.jsonl' \
    --include='run_metadata.json' --include='model_summary.json' --include='selection.json' \
    --include='final_evaluation.json' --include='summary.json' --include='complete.json' \
    --include='training_source.json' --include='evaluation_source.json' --include='per_tile.csv' \
    --include='*_per_tile.csv' --include='final_*.png' --include='events.out.tfevents.*' --exclude='*' \
    "$benchmark_host:$source_dir" "$benchmark_root/.source-cache/$cohort/"
done
printf 'Saved evidence snapshot. No training or evaluation jobs were changed.\n'
