# Third-Party Sources

HumanOmniV2 supplies the frozen text/audio backbone. Obtain its original weights
and software through the [source paper](https://arxiv.org/abs/2506.21277) and
follow the original project's terms. Encoding settings are recorded in
`configs/encoder_*.json`.

PsyPath-GFN builds on the GFlowNet framework and trajectory balance. The paper
attributes these methods to *GFlowNet Foundations* (JMLR, 2023) and
*Trajectory balance: Improved credit assignment in GFlowNets* (NeurIPS, 2022).

The implementation uses PyTorch, NumPy, SciPy, scikit-learn, Transformers and
the additional packages listed in `requirements-core.txt` and
`requirements-encoder.txt`. Dependencies retain their respective licenses.

The [MIT License](LICENSE) covers the authors' original code in this repository.
Dataset access and backbone-weight terms are governed by their respective
providers. Data, third-party repositories and backbone weights are obtained
separately.
