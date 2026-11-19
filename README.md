# Virtual Staining Benchmark

[Open the benchmark](https://thomasdekerf.github.io/virtual-staining-benchmark/)

A static benchmark for delivered virtual staining models across ORION CRC, BCI, MIST, IHC4BC, Finland mouse prostate and DermaRepo human skin.

The first view is a visual research atlas: 304 historical run records, 34 historical model families plus the new BBDM campaign, seven datasets represented, 570 scored evaluation rows and 3,654,537 recorded scalar measurements. Counts describe different units; they are not counts of independent experiments or claims that every run finished.

Seven interactive views are available:

- [Coverage atlas](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview): model-family × dataset coverage, record counts, dataset bars and recorded compute. Recommended opening view for communicating the scale of the research.
- [Learning explorer](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/curves): reach all 283 records with scalar histories using dataset, stain, exact training scope and metric filters. Select any number of runs, compare overlays or individual panels, inspect checkpoints, switch to logarithmic values or relative observed progress, and export CSV. Pooled validation histories retain their pool identity. Unverified raw position units remain explicit and separate.
- [Radar profiles](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/radar): choose any number of models across 31 matched cohorts. Overlay profiles or show one radar per run, with shared cohort scales and configurable metric axes. Only common full-coverage finite metrics become radar axes. Raw values and zoomable dot plots remain available; radar area is not an overall ranking.
- [Scatter explorer](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/scatter): choose any two recorded dimensions, including quality, wall time and GPU hours. Presets cover pixel/perceptual, compute/quality, embedding and nucleus trade-offs. Optional bubble area, log horizontal scale, exact point inspection and a direction-aware Pareto frontier expose trade-offs without claiming significance.
- [Metric heatmap](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/heatmap): compare all matched runs across chosen quality measures, sorted by a metric, with direction-aligned colour and values in original units.
- [Parallel coordinates](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/parallel): follow each model across selected metric axes; focus a profile to isolate it. Missing measurements remain gaps.
- [Metric relationships](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/correlation): inspect pairwise Pearson correlations between recorded run means, with each cell's sample count. Constant metrics and insufficient pairs remain undefined. These are run-level associations within a fixed cohort, not tile-level relationships or causal evidence.

The [searchable, sortable benchmark table](https://thomasdekerf.github.io/virtual-staining-benchmark/#/benchmark) remains available. Dataset, target stain, split and evaluation protocol define a comparison group. Best values are bold and second-best values are underlined, based on unrounded distinct values. Every metric has an explanation. Run pages expose objectives and validation curves, retained ground-truth/prediction pairs, actual per-tile violin plots, training time, configuration and source fingerprints. Up to three runs can be compared side by side.

Historical result scores retain their 15 October snapshot. Separately dated November campaign cards show the eleven HER2Match tasks, numerical recoveries, and HAPS/PaPIS readiness without promoting unfinished work into scored results. These cards are saved evidence, not a live cluster connection. The small `dist/data/visual-summary.json` records its source fingerprints and scope notes.

## Evidence and coverage

The snapshot reconciles every model-priority entry and ORION run row in `Virtual_Staining_Model_Priorities.xlsx`, all 130 completed breast runs, all 360 breast dataset/split result rows, the separate 24-patch pathology pilot, and saved cluster runs absent from the workbook. Deferred and excluded methods retain their reasons without invented scores. Details and the workbook fingerprint are in `dist/data/reconciliation.json`.

Held-out results, selected-checkpoint validation, latest training measurements and pilot evaluations stay separate. Legacy and matched FP32 scikit-image protocols are also separate. Unstained results enter the ranking only from `complete.json`, after the core profile and StarDist, StarDist 2×, HoVerNet and Cellpose-SAM evaluations finish. Metrics still outside this profile remain explicitly pending. A single numeric advantage is not statistical superiority, and image similarity does not establish diagnostic equivalence.

Breast production previews and per-tile outputs were moved to an Ubuntu archive that was inaccessible for this snapshot. Their complete metric means and validation trajectories are preserved. Unavailable assets are labelled in the run view. Training checkpoints, raw source data, private paths and subject identifiers are not distributed here. Retained field images are derived benchmark previews. Original datasets and third-party model materials retain their own terms.

## Preview

No frontend dependencies or build step are required:

```sh
python3 -m http.server 4173 --directory dist
```

Open `http://localhost:4173/`. Hash routes work under a GitHub Pages project path without server routing.

## Refresh a snapshot

The site does not poll private servers or imply that a saved training state is live. To update it, collect evidence on a machine that already has access to the training workspace and cluster:

```sh
python3 -m venv .build-env
.build-env/bin/pip install -r tools/requirements.txt
bash tools/sync_evidence.sh
.build-env/bin/python tools/build_data.py --workspace /path/to/VirtualStaining
.build-env/bin/python tools/build_visual_summary.py --workspace /path/to/VirtualStaining
.build-env/bin/python tools/validate_data.py
```

First refresh the workspace campaign progress snapshots and Excel from completed evaluation reports. The synchronizer only reads allowed result files into the ignored `.source-cache/` directory. It does not copy model weights or change jobs. `BENCHMARK_SSH_HOST` and `BENCHMARK_REMOTE_ROOT` can override its source. Completed unstained scores, sample counts and checkpoint/manifest fingerprints are reconciled with Excel. Workbook mismatches stop export rather than silently replacing values. Excel is never edited by the exporter.

`--quick` skips TensorBoard/media extraction for a first preview. `--reuse-curves` may be used after presentation or exporter changes **only if the raw TensorBoard snapshot is unchanged**. It reuses the full exported scalar CSVs, not the downsampled chart data. Dense display curves contain at most 700 points; full measurements remain downloadable.

Review and commit the refreshed `dist/` snapshot, then push to `main`. The included workflow validates static references and data invariants and deploys only `dist/` through [GitHub Pages Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). Source datasets, credentials and `.source-cache/` must stay out of Git.

## Metric interpretation

Protocol definitions, sample coverage, means and missing measurements are preserved. Violin density is estimated from every finite per-tile observation, with Gaussian kernels and Silverman bandwidth (minimum range/200); its width is normalized within each violin. It depicts tile heterogeneity, not independent training-seed uncertainty. Group bootstrap intervals, where recorded, concern source groups. Detector agreement uses automatic reference detections, not manually annotated cells. Detailed definitions are available in the site glossary.
