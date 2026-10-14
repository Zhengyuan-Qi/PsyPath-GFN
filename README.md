# PsyPath-GFN

**Psychometric-Supervised Adaptive Evidence Composition for Multimodal Mental Health Screening**

Implementation of PsyPath-GFN for depression screening on DAIC-WOZ and E-DAIC,
and PTSD screening on E-DAIC. HumanOmniV2 encodes text and audio; aligned numeric
visual descriptors form the visual evidence view. The model generates evidence
combinations, estimates severity and screening risk with PIM, and averages
predictions across 48 paths.

## Install

Use Linux, Python 3.10 and a CUDA-capable NVIDIA GPU. Online encoding checks for
at least 30 GiB free GPU memory; the downstream model runs on CPU. The recorded
environment uses PyTorch 2.11.0 with CUDA 13.0 and a compatible NVIDIA driver.

    git clone https://github.com/Zhengyuan-Qi/PsyPath-GFN.git
    cd PsyPath-GFN
    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-encoder.txt
    python -m pip check

For downstream-only work, install requirements-core.txt instead.

## Prepare Data And Weights

Obtain DAIC-WOZ/E-DAIC from their data owners and HumanOmniV2 weights through the
[original project](https://arxiv.org/abs/2506.21277). The original release format
is required: transcripts, audio and time-aligned numeric visual descriptors.
The repository contains source code, configurations and aggregate reference
metrics. Clinical data and model weights are obtained or generated separately.

Follow the [reproduction guide](docs/REPRODUCIBILITY.md) to extract representations
and train the downstream model. After reproduce.py main has finished, export a
development-selected checkpoint for each task:

    python prepare_weights.py \
      --runs-root runs/main --cache /absolute/CACHE \
      --backbone /absolute/HumanOmniV2 --seed 7 --output weights

Use the same backbone directory as during extraction. Export preserves the fitted
parameters and creates weights/manifest.json with content hashes. Checkpoints
contain serialized Python objects; load them only from trusted sources.

## Online Inference

The application entry point is `predict.py`. It reads an interview, encodes its
events with frozen HumanOmniV2, processes visual descriptors, and applies evidence
generation, PIM and multi-path averaging. Its inputs are interview data and weights.

Edit examples/interview.json with your authorized dataset location, participant
ID and task, then run:

    CUDA_VISIBLE_DEVICES=0 python predict.py \
      --input examples/interview.json --model-dir weights \
      --backbone /absolute/HumanOmniV2 --output results/prediction.json

Tasks are daic_depression, edaic_depression and edaic_ptsd. Output includes risk
probability, the development-selected threshold, screening decision, severity,
path-probability SD and selected event-view entries. Indices are zero-based.
Path SD describes variation across sampled predictions. The application uses one
task checkpoint; paper statistics cover five independently trained seeds. The
intended use is screening research.

## Experiments

Main comparisons and ablations use shared representations precomputed with
frozen HumanOmniV2. Standardisation and projections are fitted on training data;
model selection and thresholds use development data. Test targets are reserved
for final evaluation. Online inference recomputes the representations from the
interview and applies the fitted downstream model.

[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) gives commands for PsyPath-GFN,
component ablations, matched controls and mechanism statistics. Seeds are
7, 17, 27, 41 and 53. Configuration files and reference/ identify the protocol.
This repository covers the proposed model and its direct controls; the paper's
Supplementary Section S4 describes the external architecture comparisons.

## Tests

    python check_release.py
    python -m unittest tests.test_release_features

These checks run without clinical data or model weights. See
[docs/VALIDATION.md](docs/VALIDATION.md) for additional tests and their scope.

## License And Attribution

Original project code is under the [MIT License](LICENSE).
[THIRD_PARTY.md](THIRD_PARTY.md) records upstream versions and attribution.
Third-party software, data and backbone weights retain their own terms.
