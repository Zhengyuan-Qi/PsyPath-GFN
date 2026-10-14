# Training And Evaluation

## 1. Authorized data and label contract

Obtain DAIC-WOZ and E-DAIC through their data owners and comply with the applicable
access terms. Obtain the HumanOmniV2 weights separately. Do not redistribute any of
these assets with this code. Keep original labels and recordings read-only.

Inspect the preparation interface:

```bash
python scripts/build_clinical_label_contract_v1.py --help
python -m scripts.run_meter_mh_cache --help
```

Example extraction (replace all capitalized paths):

```bash
python scripts/build_clinical_label_contract_v1.py --daic-root DAIC_ROOT \
  --edaic-root EDAIC_ROOT --output-dir CONTRACT
python -m scripts.run_meter_mh_cache daic --daic-root DAIC_ROOT \
  --edaic-root EDAIC_ROOT --meld-source-rows unused.json --contract-dir CONTRACT \
  --model HUMANOMNIV2_WEIGHTS --output CACHE/daic --device cuda --threads 8 \
  --batch-size 1 --pad 0.4 --max-seconds 12
```

Repeat the second command with `edaic` and `--output CACHE/edaic`. Preserve the
prompt definitions and response-event alignment rules used by the encoder.

The label-contract builder validates source hashes, item masks and score bounds.
Depression uses `depression_phq8_screening`; PTSD uses `ptsd_dsm_iv_binary`.
Missing items retain their masks. Each task uses its official split independently;
joint-release experiments require the canonical split for overlapping participants.

Cache layout is `CACHE/{daic,edaic}/cache/PID.npz`, with the extraction contract
in each release directory. Each participant archive contains C/A/Q/V arrays and
the metadata used for response-event alignment. `configs/encoder_*.json` records
the extraction settings. Run the encoder with the authorized dataset roots and
backbone path from `--help`; the loader requires the actual release files. C/A/Q
are text/audio-conditioned representations. V uses aligned numeric descriptors,
not raw video frames. Standardization and projection are fitted on training
participants during downstream fitting.

The shared encoder interface requires `--meld-source-rows`; its placeholder is
unused for DAIC-WOZ and E-DAIC. Extraction records model and prompt fingerprints.

## 2. Main model and direct controls

```bash
python reproduce.py main --cache CACHE --contract CONTRACT --output runs/main
python reproduce.py no-pim --cache CACHE --contract CONTRACT --output runs/no_pim
python reproduce.py score-permutation --cache CACHE --contract CONTRACT --output runs/permuted
python reproduce.py reinforce --cache CACHE --contract CONTRACT --output runs/reinforce
python reproduce.py fixed-views --cache CACHE --contract CONTRACT --output runs/fixed_views
python reproduce.py pool-cls --cache CACHE --contract CONTRACT --formal-root runs/main --output runs/pool_cls
python reproduce.py pool-mtl --cache CACHE --contract CONTRACT --formal-root runs/main --output runs/pool_mtl
```

Replace CACHE and CONTRACT with absolute directories. Pooling uses the main
run's train-fitted event projections, not its learned predictor parameters.
It trains a new scaler/head. The classification head uses binary labels only;
the matched head additionally uses severity supervision with weight 0.3.
Architectures, hyperparameters, checkpoints and thresholds are selected on the
development set. Evaluation includes all five fixed seeds.

The matched structural analysis uses 36 epochs per arm. The main model uses 18:

```bash
python scripts/run_psypath_alternating_structure_formal.py --phase orchestrate \
  --meter-root CACHE --contract-dir CONTRACT --output runs/structure
```

`protocols/psypath_alternating_structure_formal_20260908.md` specifies the paired
training schedule, development selection and evaluation procedure.

## 3. External architectures

Install the dependencies in `THIRD_PARTY.md`. The paper compares ten architectures
adapted to the shared representations: `tfn lmf mult misa self_mm cubemlp dmd
simsuf cross_modal_attention` and GsiT.

```bash
python scripts/run_published_models_on_meter_cache.py \
  --meter-root CACHE --contract-dir CONTRACT --multibench-root third_party/MultiBench \
  --external-models-root third_party --output runs/external \
  --models tfn lmf mult misa self_mm cubemlp dmd simsuf cross_modal_attention \
  --device cpu --threads 8
```

`configs/external_*.json` contains per-architecture settings, seeds, epochs,
weight decay and stopping policy. The command above shows the runner interface;
the launcher below applies the recorded configurations. `${PROJECT}`,
`${THIRD_PARTY}` and `${EXTERNAL_RESULTS}` are directory placeholders.
Architectures retain their respective input processing and parameter counts.

The convenience launcher applies these settings to all three tasks:
`python reproduce_external.py --cache CACHE --contract CONTRACT --third-party
third_party --output runs/external --dry-run`. Remove `--dry-run` to execute.
The launcher uses CPU and eight threads with the recorded model settings.

For GsiT, use the selected author commit and set the following environment paths:

```bash
export PSYPATH_CACHE=/absolute/CACHE
export PSYPATH_CONTRACT=/absolute/CONTRACT
export PSYPATH_GSIT_UPSTREAM=/absolute/third_party
export PSYPATH_GSIT_OUTPUT=/absolute/runs/gsit
python gsit/run_suite.py
```

GsiT retains both development-F1 and development-BCE selections. The former is
primary; the latter is the selection sensitivity check. All 15 development locks
are written before test evaluation. `configs/gsit.json` records the protocol.

## 4. Mechanism analysis and statistics

The cross-fitted deletion audit ranks events by model-output changes on one
sample group and evaluates deletion on the other. Its selection does not use
test targets. Run the following for each task and each of the five seeds:

```bash
python scripts/audit_psypath_path_utility_fast.py \
  --meter-root CACHE --contract-dir CONTRACT --task daic_depression \
  --checkpoint runs/main/daic_depression/checkpoints/daic_depression/gflownet/seed_7.pt \
  --split test --samples 32 --random-controls 8 --skip-path-utility \
  --threads 8 --device cpu --output runs/deletion/daic_depression_seed_7.json
```

For the path-budget curves, extract all 48 locked draws without refitting:

```bash
for task in daic_depression edaic_depression edaic_ptsd; do
  for seed in 7 17 27 41 53; do
    python scripts/audit_psypath_v47_locked.py \
      --task "$task" --seed "$seed" --cache CACHE --contract CONTRACT \
      --checkpoint "runs/main/$task/checkpoints/$task/gflownet/seed_$seed.pt" \
      --out runs/path_draws --skip-timing
  done
done
python scripts/summarize_multipath.py \
  --input-dir runs/path_draws --output runs/multipath_summary.json
```

At K=1,2,4,8,16,48, the same 48 draws are partitioned into nonoverlapping
consecutive groups of K. Brier loss and severity MAE are averaged over groups
and participants before computing the mean and sample SD over five seeds.
Relative reductions are computed within each seed against its K=1 loss and
then averaged. Matched-random paths preserve event counts and view-bundle
multisets. The aggregate schema matches reference/multipath_summary.json.
Unknown scores are excluded from severity error, never filled with zero.

Raw draw files contain participant-level data and stay in the authorized
research environment. The summary contains only seed-level aggregates and
input hashes. Optional CPU timing omits HumanOmniV2 encoding, I/O and
preprocessing; omit --skip-timing only when measuring that downstream scope.

Paper tables use all five seeds and sample SD (`ddof=1`), computed from per-seed
records with `summarize.py`. Some runner outputs also contain population SD
(`ddof=0`); these fields use a different denominator. Paired intervals describe
seed-level effects. Deletion measures prediction dependence, permutation tests
participant-score correspondence, and grouped averaging evaluates aggregation.

## 5. Training To Online Inference

The main runner saves development-selected parameters under
runs/main/TASK/development_checkpoints/TASK/gflownet/seed_SEED.pt. The test
checkpoint uses the same selected parameters and adds evaluation provenance.

```bash
python prepare_weights.py --runs-root runs/main --cache CACHE \
  --backbone HUMANOMNIV2_WEIGHTS --seed 7 --output weights
python predict.py --input examples/interview.json --model-dir weights \
  --backbone HUMANOMNIV2_WEIGHTS --output results/prediction.json
```

Use an absolute backbone path from extraction onwards, and do not relocate or
touch backbone files before export: the extraction contract records their paths
and timestamps. Export creates portable content hashes for subsequent inference.
The exporter retains train-fitted projections, PIM, policy parameters and
development thresholds, with one fixed seed shared across tasks. It verifies
policy tensors and synthetic PIM outputs after serialization. To deploy another
paper seed, repeat export with --seed and a new output directory.

Only load trusted checkpoint files. They contain serialized Python/sklearn
objects; matching hashes establish file identity, not the trustworthiness of
an unknown source. Data and weights are obtained or generated separately.
Release checks are documented in VALIDATION.md.
