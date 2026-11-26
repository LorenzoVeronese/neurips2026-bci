# Software Permission Notice & Public Asset Attributions (`neurips-bci`)

## 1. Open-Source Permission for `neurips-bci`

Copyright (c) 2026 NeurIPS BCI Contributors.

Anyone is free to use, copy, modify, merge, publish, and distribute the original source code, configurations, documentation, and interactive visualizations in this repository for research, educational, or commercial work, provided that this attribution notice is kept with substantial copies and that downstream users understand the software is supplied without any warranty or liability.

## 2. Plain-Language Terms Used Below

- **Electroencephalography (EEG)**: scalp-electrode recordings of electrical brain activity.
- **Brain-computer interface (BCI)**: software translating EEG signals into discrete computer commands.
- **Motor imagery (MI)**: mentally rehearsing a limb movement without muscle contraction.
- **Mental subtraction (SUB)**: silently subtracting numbers in working memory.
- **Word generation (WG)**: silently generating words in response to a cue.
- **Foundation model**: a neural network pretrained on large external EEG collections prior to task adaptation.
- **Creative Commons Attribution 4.0 (`CC-BY-4.0`)** and **Open Data Commons Attribution 1.0 (`ODC-BY-1.0`)**: open data licenses permitting redistribution and derivative model training when the original dataset creators are credited.
- **Berkeley Software Distribution 3-Clause (`BSD-3-Clause`)** and **Massachusetts Institute of Technology (`MIT`)** licenses: permissive open-source software and model-weight licenses requiring preservation of copyright attribution.

## 3. Public EEG Corpora Used During Training

All training and validation recordings come exclusively from publicly accessible, redistributable human EEG datasets whose original open licenses remain in force:

| Corpus Role in `neurips-bci` | Public Repository Identifier | Tasks & Montage | License | Attribution Summary |
|---|---|---|---|---|
| Primary Track 2 BCI Corpus (`PROTEUS`) | `NEMAR nm000290` (`v1.0.0`) | 10 public participants (`P01..P10`), 14 sessions, 41 EEG channels, `MI` / `SUB` / `WG` across `Graz` and `BrainHero` interfaces (`2,655` 4-s windows) | `CC-BY-4.0` | Public training release of the 2026 NeurIPS EEG Challenge Track 2 hosted on the OpenNeuro / NEMAR archive |
| Auxiliary Motor Imagery Corpus (`Dreyer2023`) | `Zenodo 8089820` | Multi-participant left/right hand `MI` EEG resampled and interpolated to the 41-channel 120 Hz target montage | `CC-BY-4.0` | Public multi-subject motor imagery database archived on Zenodo record `8089820` |
| Auxiliary Multi-Task Corpus (`Shin2018`) | `DepositOnce 5830.2` | 26-participant cued `WG` and resting baseline EEG mapped via 3D spherical-spline interpolation to the 41-channel montage | `CC-BY-4.0` | Public TU Berlin DepositOnce archive `10.14279/depositonce-5830.2` |
| Auxiliary Mental Arithmetic Corpus (`Zyma2019`) | `PhysioNet eegmat` (`v1.0.0`) | 19-channel `SUB` and resting-state EEG mapped via 3D spherical-spline interpolation to the 41-channel montage | `ODC-BY-1.0` | Public PhysioNet EEG During Mental Arithmetic Tasks database (`eegmat` v1.0.0) |

## 4. Public Pretrained EEG Checkpoints Used by Foundation Organs

All foundation-model encoders shipped in compressed pure-NumPy `float16` format (`base_eegpt_fp16.npz`, `codebrain_41.npz`, `cbramod_41.npz`, `mirepnet_fp16.npz`) originate from permissive open-source HuggingFace repositories hosted under the `braindecode` organization:

| Organ in H10 | Public Checkpoint Identifier | Role in H10 Pipeline | Upstream License |
|---|---|---|---|
| `EEGPT-19` | `HuggingFace braindecode/eegpt-pretrained` | Frozen encoder features compressed via rank-64 singular value decomposition for 19 causal linear probes (`lat52`, `rms56`, `clean4`, `rms4c`) | Permissive open-source (`MIT` / `BSD-3-Clause` model wrapper) |
| `CodeBrain + LoRA8` | `HuggingFace braindecode/codebrain-pretrained` | Frozen encoder paired with a 2-seed 4-corpus rank-8 low-rank adapter (`weight = 0.15`, gated by `G1`) | Permissive open-source (`BSD-3-Clause` / `MIT`) |
| `CBraMod` | `HuggingFace braindecode/cbramod-pretrained` | Criss-cross spatial-temporal EEG encoder pretrained on the public Temple University Hospital EEG Corpus (TUEG, `weight = 0.10`, gated by `G1`) | `BSD-3-Clause` |
| `MIRepNet` | `HuggingFace braindecode/mirepnet-pretrained` | Motor-imagery representation network pretrained strictly on public `MI` corpora (`weight = 0.10`, gated by `G1`) | `MIT` |

## 5. Third-Party Python Libraries

`neurips-bci` builds on permissive open-source scientific Python packages declared in `pyproject.toml` and `environment.yml`, including `NumPy`, `SciPy`, `scikit-learn`, `PyTorch`, `MNE-Python`, `MNE-BIDS`, `MOABB`, `pyRiemann`, `Braindecode`, `Benchopt`, `NeuralSet` / `NeuralFetch` / `NeuralTrain` / `NeuralBench`, `nemar-py`, `huggingface_hub`, and `safetensors`. Each library remains governed by its own upstream open-source license.
