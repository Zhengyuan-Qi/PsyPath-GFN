# Tests And Reference Checks

Run the lightweight release tests:

    python check_release.py
    python -m unittest tests.test_release_features -v

With requirements-core.txt installed, run the existing numerical/model tests:

    python check_package.py
    python test_online.py

## Scope

| Check | Coverage |
| --- | --- |
| Archive integrity | Python/JSON parsing and SHA256 checks |
| Data-free tests | Input contracts, portable paths, command construction and path aggregation |
| Saved-result checks | F1/AUROC for 15 task/seed records; multi-path losses and sample SDs |
| Checkpoint replay | DAIC-WOZ seed 7, all 47 test participants |
| Fresh-encoding comparison | One complete DAIC-WOZ training interview; probability, severity and selections matched the saved-representation route |

The checkpoint and fresh-encoding checks were run in the research environment.
`prepare_weights.py` checks tensors and synthetic PIM predictions during export.
The data-free tests exercise the code interfaces; clinical evaluation uses the
authorized data and trained models described in [the experiment guide](REPRODUCIBILITY.md).
`reference/` contains seed-level aggregate metrics.
