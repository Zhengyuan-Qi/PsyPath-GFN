# Third-Party Sources

- HumanOmniV2: obtain authorized original weights separately; source paper
  https://arxiv.org/abs/2506.21277 . Encoding settings are in `configs/encoder_*.json`.
- MultiBench: https://github.com/pliang279/MultiBench , commit
  `49f9be224f342de005b7f001281f33df4301eb22`.
- GsiT: https://github.com/drewjin/GsiT , commit
  `76332ba433ea93fba1b2ebe9f1c85587f5b99a13`;
  paper https://aclanthology.org/2025.acl-long.109/ . Place under `third_party/GsiT`.
  The adapter selects the author's dense PyTorch attention backend. It does not
  introduce a new graph attention algorithm.
- Self-MM: https://github.com/thuiar/Self-MM , commit
  `1786283c81eeb507f317fa1c70a3faf77e67cee0`.
- The remaining architecture adaptations record their papers and source versions
  in `configs/external_*.json` and source headers. These are adaptations to
  the paper's shared representations, not the original benchmark encoders.

Dependency repositories, their datasets and their weights are not bundled.
Install the two runtime source dependencies at the recorded commits:

```bash
git clone https://github.com/pliang279/MultiBench third_party/MultiBench
git -C third_party/MultiBench checkout 49f9be224f342de005b7f001281f33df4301eb22
git clone https://github.com/drewjin/GsiT third_party/GsiT
git -C third_party/GsiT checkout 76332ba433ea93fba1b2ebe9f1c85587f5b99a13
```

MIT notices for the recorded MultiBench, GsiT, Self-MM and DMD sources are retained
in `third_party_notices/`. They continue to apply to the corresponding material.
The project MIT license covers the authors' original code and adaptations; it
does not replace another project's license or authorize redistribution of data.

SIMSUF is implemented as a standalone project architecture adaptation in
`scripts/published_simsuf_adapter.py`, with paper and upstream commit attribution.
Its upstream repository is https://github.com/Yan-LiJI/SIMSUF at
`ee823bf94335b280177030ea2715dcac2e6b3e23`. That checkout has no license file
and is not redistributed. This release does not assign a license to its source.
