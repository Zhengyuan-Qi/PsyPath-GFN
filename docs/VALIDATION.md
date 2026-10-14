# Tests And Reference Checks

Run the release-interface checks:

```bash
python check_release.py
python -m unittest tests.test_release_features -v
```

With `requirements-core.txt` installed, run the model and statistics tests:

```bash
python check_package.py
python test_online.py
python -m unittest scripts.test_psypath_review_controls -v
python -m unittest scripts.test_review_statistics -v
python -m unittest tests.test_score_isolation -v
```

These tests cover input contracts, checkpoint hashes, portable paths, command
construction, split-specific target loading, matched controls and aggregation.
They use synthetic inputs and run without clinical recordings or model weights.

`reference/` contains aggregate results for the five paper seeds, including
matched-control differences and paired participant intervals.
`verify_saved_results.py` and `summarize.py` check saved experiment outputs.

Checkpoint replay in the research environment covered DAIC-WOZ seed 7 and all
47 test participants. A fresh-encoding comparison on one complete DAIC-WOZ
training interview matched the saved-representation route in probability,
severity and selected entries. `prepare_weights.py` verifies model tensors and
synthetic PIM predictions during checkpoint export.

Clinical evaluation uses authorized data and trained checkpoints as described
in [Training And Evaluation](REPRODUCIBILITY.md).
