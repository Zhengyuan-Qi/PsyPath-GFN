# Matched Evidence-Composition Controls

`scripts/run_psypath_review_controls.py` implements three training conditions:

| Condition | Measurement training | Evidence generation |
| --- | --- | --- |
| `random_pim` | Full evidence plus four random sparse sets per participant; three PIM fits | Random nonempty view bundles and events; 48 test sets |
| `score_reward` | A fixed, true-score PIM shared by the two policy conditions | GFlowNet with the explicit score-error reward term |
| `no_score_reward` | The identical fixed PIM | GFlowNet with that reward coefficient set to zero |

The paired policies share initial parameters, participant order, sampling seeds,
supervision, and training budget. Their PIM remains fixed, so this comparison
isolates the explicit score-reward coefficient within a common measurement
model. The random-PIM condition tests the full random-set training and prediction
workflow. Its event limits and view-bundle space match the generator's allowed
space; realised event counts are also recorded.

Each condition has 18 development evaluations and uses the original development
selection rule. Training uses the saved training-fitted event projections. All
15 task/seed development runs are locked before test evaluation. The reference
checkpoint supplies the event projections and configuration; the new conditions
fit their own PIM and initialise their own policies.

For the directory layout produced by the main experiment guide:

```bash
python scripts/run_psypath_review_suite.py --project . \
  --cache /absolute/CACHE --contract /absolute/CONTRACT \
  --main-root /absolute/runs/main --output /absolute/runs/evidence_controls \
  --workers 3

python scripts/summarize_psypath_review_controls.py \
  --controls /absolute/runs/evidence_controls \
  --main-root /absolute/runs/main --score-records /absolute/runs/path_draws \
  --output /absolute/runs/control-summary.json
```

The summary reports all five seeds and label-stratified paired participant
bootstrap intervals (10,000 resamples). Each participant resample is shared
across methods and all fitted seeds. Binary predictions use the thresholds
selected on development data. The locked score-record files use the same `rows`,
`pid`, and `score` fields as the multi-path evaluation records.

`reference/evidence_control_results.json` contains the complete aggregate results,
five-seed metrics, paired participant intervals, and input-file hashes. It contains
no participant identifiers or individual predictions. The reported comparisons
include weighted and positive-class F1, AUROC, Brier score, sensitivity,
specificity, severity MAE, and CCC.
