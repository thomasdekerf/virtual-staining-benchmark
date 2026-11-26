# Virtual Staining Benchmark

[Open the benchmark](https://thomasdekerf.github.io/virtual-staining-benchmark/)

A static benchmark for delivered virtual staining models across ORION CRC, BCI, MIST, IHC4BC, Finland mouse prostate, DermaRepo human skin, HER2Match and HEMIT.

The 26 November 2026 snapshot contains **300 public run records and 622 ranked
dataset/split results across eight datasets**. All 596 previously published
ranked rows retain their exact means, checkpoints, manifests and comparison
protocols. The 26 new rows come from six HEMIT methods, HER2Match HAT, Finland
BBDM and five BCI methods. Unfinished training remains unranked.

The first view is a visual atlas of reviewed, paper-backed methods across eight datasets. Every catalog method links to its paper; preprints are labelled. Conventional dataset adaptations remain, while unpublished local gates, fusion modules, refiners, marker-panel studies and other research recipes are omitted. The dataset bars count distinct model families with held-out test evidence, once per family. Training and planned coverage remain separate from completed test scores.

Seven interactive views are available:

- [Coverage atlas](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview): model-family × dataset coverage, record counts, dataset bars and recorded compute. Recommended opening view for communicating the scale of the research.
- [Learning explorer](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/curves): reach all available records with scalar histories using dataset, stain, exact training scope and metric filters. Select any number of runs, compare overlays or individual panels, inspect checkpoints, switch to logarithmic values or relative observed progress, and export CSV. Pooled validation histories retain their pool identity. Unverified raw position units remain explicit and separate.
- [Radar profiles](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/radar): choose any number of models across the available matched cohorts. Overlay profiles or show one radar per run, with shared cohort scales and configurable metric axes. Only common full-coverage finite metrics become radar axes. Raw values and zoomable dot plots remain available; radar area is not an overall ranking.
- [Scatter explorer](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/scatter): choose any two recorded dimensions, including quality, wall time and GPU hours. Presets cover pixel/perceptual, compute/quality, embedding and nucleus trade-offs. Optional bubble area, log horizontal scale, exact point inspection and a direction-aware Pareto frontier expose trade-offs without claiming significance.
- [Metric heatmap](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/heatmap): compare all matched runs across chosen quality measures, sorted by a metric, with direction-aligned colour and values in original units.
- [Parallel coordinates](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/parallel): follow each model across selected metric axes; focus a profile to isolate it. Missing measurements remain gaps.
- [Metric relationships](https://thomasdekerf.github.io/virtual-staining-benchmark/#/overview/correlation): inspect pairwise Pearson correlations between recorded run means, with each cell's sample count. Constant metrics and insufficient pairs remain undefined. These are run-level associations within a fixed cohort, not tile-level relationships or causal evidence.

The [searchable, sortable benchmark table](https://thomasdekerf.github.io/virtual-staining-benchmark/#/benchmark) remains available. Dataset, target stain, split and evaluation protocol define a comparison group. Best values are bold and second-best values are underlined, based on unrounded distinct values. Every metric has an explanation. Run pages expose objectives and validation curves, retained ground-truth/prediction pairs, actual per-tile violin plots, training time, configuration and source fingerprints. Up to three runs can be compared side by side.

The 19 November refresh preserves every previously published ranked result and adds the verified current results: nine completed HER2Match models, three newly completed Finland families and recovered skin ASP. This gives 19 complete Finland test profiles and 21 for skin. Separately dated campaign cards show unfinished training, evaluations and the focused 17-run expansion. The expansion adds HEMIT, ImplicitStainer and MiPHEI-ViT to the breast HER2 anchor comparisons, plus BBDM across the original breast anchors, Finland and skin. Original breast and HER2Match target 14 benchmark families; Finland and skin target 22. These are planned targets until the runs and full evaluations finish. The small `dist/data/visual-summary.json` records source fingerprints and scope notes.

## Published architecture expansion

The 26 November refresh adds the **HEMIT dataset**, separate from the HEMIT
model and ORION's reverse translation task. Six native-horizon-qualified methods
have complete 630-field validation and 945-field test results; BBDM and I²SB
retain unranked progress. All eight include full deduplicated parent/continuation
histories. Numerical DAPI/CD3/panCK channels use exact released uint8/255 storage
scaling, without physical fluorescence calibration or a patient-disjoint claim.
Channel PCC exposes its defined/total field counts. This cohort excludes RGB
Lab/DAB and clinical scores; LPIPS is an operational feature distance. CUT's poor
correspondence and Uformer's weak CD3 correspondence are explicitly flagged.
Per-field predictions and score distributions are unavailable in this snapshot.
PRECISE is user-deferred, with no active objective.

The same refresh adds the newly completed HER2Match HAT, original Finland BBDM,
and five BCI HER2 full profiles (I2I-Mamba, PD-UNIST, STABLE, HistDiST and D-VST),
while retaining all previous scored means exactly. BCI HAT and the separate
Finland BBDM plateau continuation remain unranked. A completed parent result does
not imply that its continuing training stage is final.

The 23 November expansion adds full author-capacity **Pix2PixHD**, **Uformer-B**,
**DCLGAN** and **I²SB** implementations for the HER2Match and BCI HER2 comparisons.
Their papers cover multiscale conditional GANs, restoration Transformers, dual
contrastive unpaired translation and paired diffusion bridges. The paper
catalogue now contains 46 methods; the implemented HER2Match landscape targets 34
families. These are conventional staining adaptations with pinned author code,
explicit objectives and per-dataset manifests, rather than scores reproduced
from the authors' original datasets.

Queued work carries no fabricated training updates or test scores. Each new
recipe requires its actual full-capacity four-GPU positive-LR checkpoint/resume
check and independent nine-field validation recomputation. Additional checks bind
DCLGAN's six learned networks and three optimizers, I²SB's online/EMA models and
pretrained initialization, and Pix2PixHD's frozen VGG19 feature network. Complete
validation/test measurements remain necessary for scored publication. Shared
metric/data code is explicitly reviewed before newly published scores join an
existing comparison cohort; old scientific records and gallery bindings remain
exact.

Existing MCS-Stain failures are flagged on their actual selected checkpoints:
white collapse on ORION, skin and HER2Match, and purple/cyan artifacts in Finland.
These diagnoses preserve the real historical results and distinguish completed
evaluation from useful prediction quality.

## Evidence and coverage

The complete local evidence ledger reconciles every model-priority entry and ORION run row in `Virtual_Staining_Model_Priorities.xlsx`, all 130 completed breast runs, all 360 breast dataset/split result rows, the separate 24-patch pathology pilot, and saved cluster runs absent from the workbook. Historical reconciliation totals retain that original scope. The website, gallery, downloads and visible counts contain only the reviewed published-method subset, documented in `dist/data/publication-policy.json`. Removing a public entry does not delete its original research evidence or checkpoint.

`tools/published-models.json` is the explicit paper and recipe registry. A published backbone alone does not approve a custom composite: each visible run ID must also be reviewed. New or unknown runs remain excluded until reviewed. Both snapshot exporters apply this policy, and deployment validates every public inventory, direct run JSON, CSV download and image asset against it.

Held-out results, selected-checkpoint validation, latest training measurements and pilot evaluations stay separate. Legacy and matched FP32 scikit-image protocols are also separate. Unstained results enter the ranking only from `complete.json`, after the core profile and StarDist, StarDist 2×, HoVerNet and Cellpose-SAM evaluations finish. Metrics still outside this profile remain explicitly pending. A single numeric advantage is not statistical superiority, and image similarity does not establish diagnostic equivalence.

Breast production previews and per-tile outputs remain in the Ubuntu archive outside this published snapshot; the original dataset bundle has been restored for the selective training extension. Their complete metric means and validation trajectories are preserved. Unavailable assets are labelled in the run view. Training checkpoints, raw source data, private paths and subject identifiers are not distributed here. Retained field images are derived benchmark previews. Original datasets and third-party model materials retain their own terms.

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
.build-env/bin/python tools/build_data.py --workspace /path/to/VirtualStaining --refresh-existing
.build-env/bin/python tools/build_visual_summary.py --workspace /path/to/VirtualStaining
.build-env/bin/python tools/validate_data.py
```

First refresh the workspace campaign progress snapshots and Excel from completed evaluation reports. The synchronizer only reads allowed result files into the ignored `.source-cache/` directory. It does not copy model weights or change jobs. `BENCHMARK_SSH_HOST` and `BENCHMARK_REMOTE_ROOT` can override its source. Completed unstained scores, sample counts and checkpoint/manifest fingerprints are reconciled with Excel. Workbook mismatches stop export rather than silently replacing values. Excel is never edited by the exporter.

`--quick` skips TensorBoard/media extraction for a first preview. `--reuse-curves` may be used after presentation or exporter changes **only if the raw TensorBoard snapshot is unchanged**. It reuses the full exported scalar CSVs, not the downsampled chart data. Dense display curves contain at most 700 points; full measurements remain downloadable.

Review and commit the refreshed `dist/` snapshot, then push to `main`. The included workflow validates static references and data invariants and deploys only `dist/` through [GitHub Pages Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). Source datasets, credentials and `.source-cache/` must stay out of Git.

## Metric interpretation

Protocol definitions, sample coverage, means and missing measurements are preserved. Violin density is estimated from every finite per-tile observation, with Gaussian kernels and Silverman bandwidth (minimum range/200); its width is normalized within each violin. It depicts tile heterogeneity, not independent training-seed uncertainty. Group bootstrap intervals, where recorded, concern source groups. Detector agreement uses automatic reference detections, not manually annotated cells. Detailed definitions are available in the site glossary.
