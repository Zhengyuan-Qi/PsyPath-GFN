# Matched Structural Comparison

## Comparison

This comparison jointly evaluates latent clinical routing and event/bundle
factorization against a flat conditional GFN with the same admissible actions.
Both models use shared frozen HumanOmniV2 representations and project-defined
implementations of the two policies.

Both arms alternate PIM fitting and selector optimization. PIM coefficients are
identical at cycle zero and can differ after policy-dependent refitting. The
reward definition and fitting procedure are matched. The fixed-PIM comparison
holds the numerical reward function constant and is a separate control.

## Training And Selection

Protocol fixed on 2026-09-08.

- Tasks: DAIC depression, E-DAIC depression, E-DAIC PTSD.
- Seeds: 7, 17, 27, 41, 53.
- Participant-level official splits and explicit clinical-label profiles from
  `clinical_label_contract_v1`; no source-label modification.
- Shared training-fitted event projections and common encoder/log-Z initialization.
- Hidden width 64 in both arms; exact parameter counts are recorded by the runner.
- Two arms: hierarchical GFN and flat joint event/bundle conditional GFN.
- Both have 15 nonempty view bundles, no repeated event, STOP after at least two
  events, maximum min(16, ceil(0.30*N)); same backward-permutation convention.
- AdamW, learning rate 3e-4, weight decay 1e-4, participant gradient
  accumulation 4, gradient clipping 1, four training trajectories per participant.
- Reward: log R = -(binary CE + 0.30 * normalized absolute
  severity error + 0.05 * selected-event fraction) / 0.50. Standard TB loss.
- Three PIM fits: at the start of each cycle, using full evidence plus
  four augmented masks per training participant and grouped out-of-fold fitting.
- Selector schedule: 6 + 6 + 24 epochs = 36 total. The first 18 reproduce the
  original 3*6 schedule; the last 18 extend only its final fixed-operator phase.
  The 36-epoch schedule is fixed across arms and seeds.
- Every epoch: Dev with 16 paths; original lexicographic selection (weighted-F1,
  macro-F1, AUROC, severity CCC, negative Brier), including Dev threshold choice.
- Test: 48 paths and one development-selected checkpoint per arm/seed, evaluated
  after all 30 development selections are locked and their hashes validated.
- This supplementary analysis was specified after the main test results were
  available; settings for both arms were fixed before their evaluation.

## Execution

The runner uses up to 30 single-thread CPU workers and 32 affinity cores in total.
Training-only preprocessing is shared between paired arms. Zero-weight auxiliary
calculations are skipped after reward and update equivalence checks. Smoke tests
use training/development participants; formal evaluation includes all five seeds.

## Validation And Statistics

Invalid labels/splits, nonfinite values, changed contracts, corrupted checkpoints
or worker failures invalidate a run. Each worker has a 45-minute timeout; timed-out
runs remain incomplete. Evaluation summaries require all specified seeds.

The primary endpoint is test weighted F1. Summaries include per-seed values, mean,
sample SD, the paired hierarchical-minus-flat effect with a 95% seed-level t
interval, and an exact two-sided sign-flip test. Secondary endpoints are macro-F1,
AUROC and severity CCC/MAE. Intervals describe variation across paired seeds.

Training diagnostics include development selection epochs and the mean TB loss
over the last six versus preceding six epochs in the final fixed-operator phase.
