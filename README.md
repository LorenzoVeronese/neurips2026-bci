# NeurIPS 2026 BCI (`neurips_bci`): Causal Electroencephalography Brain-Computer Interface Decoder

`neurips-bci` is the clean-room, object-oriented reproduction package for the **H10** causal streaming decoder developed for Track 2 of the 2026 NeurIPS Electroencephalography Challenge. **Electroencephalography (EEG)** records electrical brain activity from electrodes placed on the scalp. A **brain-computer interface (BCI)** translates those electrical signals into computer commands without requiring muscle movement.

In this benchmark, each input is one 4-second **window** of 41 scalp EEG channels sampled at 120 **hertz (Hz, cycles per second)** (`41 x 480` samples in **microvolts, uV, millionths of a volt**). Windows arrive strictly one at a time (`B = 1`) in a **causal online** stream: at step `t`, the decoder may adapt using windows `0..t` from the active recording **session** (one continuous recording visit of one participant, typically 190 to 218 windows), and all session state resets cleanly before the next session begins. For every window, the decoder predicts one of three mental commands:

- **Motor imagery (MI)**: mentally rehearsing a hand movement without contracting muscles. Imagining a hand movement triggers **event-related desynchronization (ERD)**, a drop in 8 to 30 Hz **sensorimotor rhythm (SMR)** power over the opposite side of motor cortex (for example electrode `C4` during left-hand imagery).
- **Mental subtraction (SUB)**: silently and repeatedly subtracting a fixed number from a starting number in working memory, which recruits fronto-parietal oscillations.
- **Word generation (WG)**: silently thinking of words (such as words beginning with a prompted letter), which shifts fronto-temporal and parietal oscillatory power.

## Headline Benchmark Results

Decoding performance is measured by **cell-mean balanced accuracy**: within each `(session, interface)` cell (`Graz` bar feedback or `BrainHero` game feedback), the scorer averages per-class recall (the fraction of true windows of each command decoded correctly), then averages equally across all cells (chance is `0.333`). Offline evaluation uses 16-fold **cross-validation (CV)** across the 10 public participants (`P01..P10`): 10 **leave-one-subject-out (LOSO)** folds testing unseen people, 4 **cross-session (XS4)** folds testing later recording days (`P04/S2`, `P04/S3`, `P10/S2`, `P10/S3`) of seen participants, and 2 single-session-known folds (`XS1`). Our primary offline headline **`k05`** equals `0.5 * mean(LOSO) + 0.5 * mean(XS4)`, evaluated both in recorded chronological order (**`O0`**) and across five random within-session window shuffles (**`OW5`**).

| Package / Configuration | Local 16-Fold `O0` `k05` | Local 16-Fold `OW5` `k05` | Sealed Competition Score | Worker Runtime (`4,800 s` cap) | Key Operational Property |
|---|---:|---:|---:|---:|---|
| Upload 5 (`DualEEGPT` baseline) | `80.15%` | `76.27%` | `0.8069` | `2,717.58 s` | Prior 40-network CNN + DualEEGPT stack |
| **Submission 6 (`h10_r35_submission6`)** | **`82.66%`** | **`81.35%`** | **`0.8230`** (banked) | `3,175.05 s` | Full H10 multimodal causal stack |
| **Submission 7 (`h10_r45_submission7`)** | **`82.66%`** | **`81.35%`** | **`~0.830`** (projected) | `~2,819 s` (at `2.65x` calibration) | `0` Tier-0 decision flips vs `r35`, `-27%` per-window latency (`131.0 ms -> 97.2 ms`), relaxed pace guard |

Interactive figures exploring the 41-electrode scalp montage, the 12-stage H10 signal flow, the pace-guard load-shedding simulator, and 16-fold cross-validation results are self-contained in `docs/index.html` (open directly in any browser with zero network dependencies, or serve via GitHub Pages).

---

## H10 Signal-Flow Architecture

Why does H10 combine small filter-bank networks, covariance geometry, and pretrained models? Scalp EEG differs substantially across individuals because skull thickness, cortical folding, and electrode impedance rescale and mix cortical sources differently on every head. Small domain-matched filter-bank networks adapt rapidly inside a single 200-window session when paired with covariance whitening, whereas diverse covariance and pretrained encoders contribute partially independent errors that reduce variance when fused in log-probability space.

```text
  Raw EEG Window x_t  (B=1, C=41 channels, T=480 samples; 4.0 s at 120 Hz in uV)
     |
     v
  [G] PaceGuard (tau_t in {0, 1, 2, 3})
     |   Tracks per-session time credit c_t and global run elapsed time E_t
     |   to guarantee completion below the 4,800 s competition worker timeout.
     v
  [P] Sanitize & Zero-Mean  (x_clean)
     |   Replaces non-finite samples with 0 and subtracts each channel's window mean.
     v
  [A] ArtifactGuard  (x_guard, ok_t in {0, 1})
     |   Computes robust amplitude and log-determinant z-scores against causal session
     |   history; clips samples beyond 32 robust scales and blocks flagged windows
     |   (ok_t = 0) from corrupting online adaptation buffers.
     |
     +-----------------------------------+
     |                                   |
     v                                   v
  [E] 6-Band + Broadband Covariances  {C_t^(b)}  &  PseudoChrono Alignment  (w_t = R_t^(-1/2) x_guard)
     |   Estimates shrunk 41x41 spatial covariances in 4-8, 8-13, 13-20, 20-30,
     |   30-45, and 1-45 Hz bands plus broadband. Weights past clean windows (s < t)
     |   by Gaussian similarity K_{t,s} in a 6-D non-motor drift subspace so whitening
     |   remains accurate even when windows arrive in shuffled order (OW5).
     |
     +-----------------------------------+-----------------------------------+
     |                                                                       |
     v                                                                       v
  [C] CNN40 Dual-Block Ensemble  (u_cnn40)                                [F] Orthogonal Multimodal Organs
     |   Two independent 80-net blocks (160 convolutional nets total):       |   - EEGPT-19 (u_lat52, u_rms56): 19 linear/SVD
     |   - 20 ShallowConvNets (k=25) + 20 FBCNets per block                  |     probes on frozen EEGPT, Cholesky-refit at 66, 126
     |   - 20 distilled FBModNets + 20 distilled ShallowConvNets (k=49)      |   - TSR4 (u_tsr4): multi-band sub-window tangent head
     |   - PseudoChrono adaptive BatchNorm + de-clustered Tent (t < 130)     |   - RSNSE (u_rsnse): Riemannian subspace MLP ensemble
     |   - Multimodal-teacher self-training + AG3 head refit at t = 66, 126  |   - ANCHOR-MM (r_dm, r_pr): GPU float64 dispersion
     |   - Tangent-space logistic regression + live session CSP+LDA          |     & prototype covariance residual organ
     |   - Per-participant covariance specialists + ratio class balancer      |   - CodeBrain + Rank-8 LoRA (u_cd, weight 0.15)
     |                                                                       |   - CBraMod (u_cb, weight 0.10)
     |          +------------------------------------------------------------+   - MIRepNet (u_mr, weight 0.10)
     |          |                                                                - S3D (u_s3d, weight 0.12): 16-D Fisher anchored ridge
     v          v
  [M] Causal Log-Probability Fusion & Cell Balancing  (p_t in Delta^3)
     |   1. Trajectory A (weight 0.40): 0.75 CNN40 + 0.06 lat52 + 0.12 rms56 + 0.07 TSR4 -> ratio balancer
     |   2. Trajectory B (weight 0.60): (0.90 CNN40 + 0.05 lat52 + 0.05 rms56) + 0.08 RSNSE -> Kalman balancer
     |   3. Base blend p_base = 0.40 TrajA + 0.60 TrajB, refined by ANCHOR-MM residuals (r_dm, r_pr)
     |      (also emits multimodal self-training teacher pseudo-labels back to [C] and EEGPT at t = 66, 126)
     |   4. Non-self-trained ShallowConvNet anchor (+0.06 Shallow_STOFF) to damp late-session drift
     |   5. PseudoChrono Sinkhorn-Knopp cell balancer (tau=1.3, 4 iterations) equalizing class marginals
     |   6. Causal G1 excess-agreement reliability gate scaling CodeBrain, CBraMod, and MIRepNet log-odds
     |   7. Online S3D 16-D Fisher subspace anchored ridge log-odds (+0.12)
     v
  Predicted Command y_hat_t = argmax_c p_{t,c}  in {0: MI, 1: SUB, 2: WG}
```

### Component-by-Component Signal Flow

1. **Pace Guard (`[G]`, symbol $\tau_t \in \{0, 1, 2, 3\}$)**: Reads per-window wall-clock elapsed time inside the current session and cumulative elapsed time across the entire evaluation run. On windows `8..39`, it measures median per-window latency; fast workers (`<= 0.30 s` in `r45`) receive initial session credit `F0 = 14.0 s` while slower workers receive `F0 = 6.0 s`, accumulating `0.29 s` per window (`0.205 s` in `r35`). Dropping below `1.2 s`, `0.6 s`, or `0.2 s` ratchets $\tau_t$ monotonically to Tier 1, Tier 2, or Tier 3; global elapsed thresholds (`3,550 s`, `4,300 s`, `4,420 s` in `r45`) provide a second run-level cap. Tier 1 freezes **test-time entropy minimization (Tent, adjusting batch normalization scale and shift parameters to reduce prediction uncertainty)** and skips `CodeBrain`, `MIRepNet`, and `S3D` while keeping **self-training (ST, mid-session weight refitting on high-confidence pseudo-labeled windows)** in `r45`; Tier 2 additionally drops `CBraMod`, `RSNSE`, `ANCHOR-MM`, secondary `EEGPT` seeds, `Sinkhorn-Knopp` scaling, and self-training; Tier 3 returns balanced `CNN40` alone.
2. **Sanitizer (`[P]`, symbol $x_{\text{clean}}$)**: Receives raw `(1, 41, 480)` EEG, replaces `NaN` or infinite entries with `0.0`, and subtracts each channel's temporal mean over the 480 samples to remove direct-current offset.
3. **Artifact Guard (`[A]`, symbols $x_{\text{guard}}, \text{ok}_t \in \{0, 1\}$)**: Computes peak amplitude and broadband covariance log-determinant statistics relative to a causal running median and median absolute deviation shrunk toward a training prior (`n0 = 10`). Windows exceeding `z = 8.0` set $\text{ok}_t = 0$ so downstream adapters do not absorb muscle or electrode-pop spikes, while extreme samples above `32.0` robust scales are soft-clipped toward `8.0`.
4. **Band Covariances and Pseudo-Chronological Euclidean Alignment (`[E]`, symbols $\{C_t^{(b)}\}, w_t$)**: Filters $x_{\text{guard}}$ into six physiological passbands (`4-8`, `8-13`, `13-20`, `20-30`, `30-45`, and `1-45 Hz`) plus broadband, estimating regularized `41 x 41` **spatial covariance** matrices (tables of pairwise electrode co-variation). **Euclidean alignment (EA)** whitens $x_{\text{guard}}$ via $w_t = R_t^{-1/2} x_{\text{guard}}$. Instead of weighting all past windows equally, H10 projects non-motor drift features into a 6-dimensional subspace and weights clean past windows ($s < t$) with Gaussian kernel $K_{t,s}$ (`tau = 1.3`), keeping whitening accurate under both chronological (`O0`) and shuffled (`OW5`) streams.
5. **CNN40 Dual-Block Ensemble (`[C]`, symbol $u_{\text{cnn40}}$)**: Feeds aligned EEG $w_t$ and band covariances into two independent seed blocks (`block_a_seed_offset = 200`, `block_b_seed_offset = 1200`), totaling 160 compact **convolutional neural networks (CNNs, neural networks that slide short temporal and spatial filters over the signal)**: per block, 20 `ShallowConvNet` (`k = 25`), 20 filter-bank `FBCNet`, 20 distilled `FBModNet`, and 20 distilled `ShallowConvNet` (`k = 49`) with 4-group session **batch normalization (BN, rescaling layer activations to zero mean and unit variance)**. During streaming, BN statistics update causally via $K_{t,s}$, a 64-window de-clustered `Tent` buffer adapts `ShallowConvNet` affine parameters up to window `130`, and at checkpoints `t = 66` and `t = 126` both blocks refit on replay + teacher-gated session windows (`threshold = 0.58`) alongside a 3-class anchor-gated linear readout (`AG3`). Each block blends its CNNs with a **Riemannian tangent space** (a flat vector projection of curved covariance matrices) logistic classifier, a session **common spatial patterns (CSP, class-discriminative spatial filters)** + **linear discriminant analysis (LDA, shared-covariance linear classifier)** head refitted at `66/126`, and per-participant covariance specialists, followed by causal class-ratio balancing.
6. **Orthogonal Multimodal Organs (`[F]`, symbols $u_{\text{lat52}}, u_{\text{rms56}}, u_{\text{tsr4}}, u_{\text{rsnse}}, r_{\text{dm}}, r_{\text{pr}}, u_{\text{cd}}, u_{\text{cb}}, u_{\text{mr}}, u_{\text{s3d}}$)**: Parallel encoders read complementary geometric and pretrained representations from the same causal window:
   - `EEGPT-19`: a frozen **foundation model** (a large neural network pretrained on external EEG) compressed via rank-64 **singular value decomposition (SVD, orthogonal variance factorization)** with 19 multi-seed linear heads (`8 lat52`, `3 rms56`, `6 clean4`, `2 rms4c`) updated at windows `66` and `126` via exact `float64` Cholesky Newton steps.
   - `TSR4` and `RSNSE`: multi-band sub-window tangent-space classifiers and a 5-seed Riemannian subspace multilayer perceptron ensemble.
   - `ANCHOR-MM`: a multi-metric covariance dispersion and prototype residual organ; in `r45` (`Submission 7`), its `float64` linear algebra executes directly on the **graphics processing unit (GPU)**, cutting whole-stack latency by `27%` with zero Tier-0 decision changes.
   - `CodeBrain` (`+ Rank-8 LoRA`, **low-rank adaptation**, a compact trainable two-matrix correction on frozen weights), `CBraMod`, and `MIRepNet`: three public pretrained EEG encoders bundled as pure-NumPy `float16` weights with zero `pickle` or network dependencies.
   - `S3D`: an online 16-dimensional Fisher subspace anchored ridge adapter combining multimodal embeddings.
7. **Causal Log-Probability Fusion (`[M]`, symbol $p_t \in \Delta^3$)**: Combines expert predictions in log-probability space. Trajectory A (`0.75 u_cnn40 + 0.06 u_lat52 + 0.12 u_rms56 + 0.07 u_tsr4`, ratio-balanced) and Trajectory B (`0.90 u_cnn40 + 0.05 u_lat52 + 0.05 u_rms56` mixed with `0.08 u_rsnse`, Kalman-balanced with drift `q = 0.002`) form `p_base = 0.40 TrajA + 0.60 TrajB`. `ANCHOR-MM` residuals (`lam_dm = 0.52`, `lam_proto = 0.13`, `c4_direct = 0.20`) refine `p_base` and generate the multimodal self-training teacher labels used at `t = 66, 126`. Adding `+0.06` un-adapted `Shallow_STOFF` log-odds prevents self-training drift, after which `PseudoChrono` **Sinkhorn-Knopp scaling** (alternating row and column rescaling over `4` iterations with `tau = 1.3, prior = 10.0`) balances session-cell class frequencies. Finally, the causal `G1` excess-agreement reliability gate (`[0.25, 1.00]`) weights `CodeBrain` (`0.15`), `CBraMod` (`0.10`), and `MIRepNet` (`0.10`), followed by `S3D` (`0.12`), yielding $\hat{y}_t = \arg\max_c p_{t,c}$.

---

## Repository Layout

The repository separates the strictly layered `neurips_bci/` Python package from configurations, automated tests, setup scripts, and interactive documentation:

```text
deliverables/
|-- neurips_bci/               # 11-layer one-way OOP Python package (1 class per file)
|   |-- validator/             # L0: ParameterTypeValidator, ArchiveSafetyTypeValidator
|   |-- operator/              # L1: CovarianceMatrixOperator, SubspaceMatrixOperator, SpectralSignalOperator
|   |-- configuration/         # L2: ExperimentRunConfiguration (sole JSON config I/O boundary)
|   |-- container/             # L3: WindowedCorpusStreamContainer, SessionWindowStreamContainer, ValidationFoldStreamContainer
|   |-- acquisition/           # L4: Public BIDS/MOABB/HuggingFace corpus & checkpoint Acquirers
|   |-- adaptation/            # L5: Causal EA, ArtifactGuard, Tent, Sinkhorn, Kalman, PaceGuard Adapters
|   |-- backbone/              # L6: ShallowConvNet, FBCNet, FBModNet, EEGPT, CodeBrain, CBraMod, MIRepNet Backbones
|   |-- organ/                 # L7: CNN40, EEGPT-19, TSR4, RSNSE, ANCHOR-MM, FM, S3D Organs
|   |-- decoder/               # L8: H10MultimodalEnsembleDecoder (Benchopt B=1 streaming API)
|   |-- evaluator/             # L9: CellBalancedAccuracy, CrossValidation, ArchiveContract Evaluators
|   `-- workflow/              # L10: StageWorkflows & CommandLineInterfaceWorkflow (`neurips-bci` CLI)
|-- configs/                   # Complete run configurations (h10_r45_submission7, h10_r35_submission6, h10_smoke_test)
|-- scripts/setup_conda_env.sh # Deterministic Conda + PyTorch environment installer
|-- tests/                     # AST architecture enforcement & unit/integration verification suites
|-- docs/index.html            # Self-contained offline/GitHub Pages interactive HTML5 + SVG explorer
|-- README.md                  # This guide
|-- DESIGN.md                  # 11-layer OOP architecture specification & dated design history
`-- NOTICE.md                  # Open-source software permission & public dataset/checkpoint attributions
```

---

## Environment Setup & End-to-End CLI Reproduction

### 1. Create the Conda Environment

Requires Python `>= 3.12`. Use the provided setup script (supports `--cpu` on machines without an NVIDIA GPU or `--prefix /path/to/env` on shared clusters):

```bash
cd deliverables
bash scripts/setup_conda_env.sh
conda activate neurips-bci
```

### 2. Run the Offline End-to-End Smoke Test & Architecture Audit

Verify 11-layer one-way import rules, class-per-file conventions, causal `B = 1` streaming, and all 6 archive survival gates in seconds on CPU without network access:

```bash
neurips-bci smoke-test --config configs/h10_smoke_test.json
pytest tests/
```

### 3. Reproduce Submission 7 (`h10_r45_submission7`) Stage by Stage or End to End

Every run reads all hyperparameters from a single **JavaScript Object Notation (JSON)** configuration file via `ExperimentRunConfiguration` and writes `resolved_run_configuration.json` (governed by `master_random_seed`) into `output_run_directory`:

```bash
# Option A: Full end-to-end pipeline (download -> extract -> train -> package -> verify -> evaluate)
neurips-bci reproduce --config configs/h10_r45_submission7.json

# Option B: Individual pipeline stages
neurips-bci download --config configs/h10_r45_submission7.json
neurips-bci extract  --config configs/h10_r45_submission7.json
neurips-bci train    --config configs/h10_r45_submission7.json
neurips-bci package  --config configs/h10_r45_submission7.json
neurips-bci verify   --config configs/h10_r45_submission7.json
neurips-bci evaluate --config configs/h10_r45_submission7.json
```

To reproduce the banked Submission 6 (`0.823` sealed) configuration instead, pass `--config configs/h10_r35_submission6.json`.

---

## Public Datasets & Pretrained Checkpoints

All training corpora and pretrained backbones are publicly redistributable and declared in `NOTICE.md`:

- **PROTEUS Track 2 Corpus (`NEMAR nm000290` v1.0.0, `CC-BY-4.0`)**: 10 participants, 14 sessions, 41 EEG channels, `2,655` 4-second windows across `MI`, `SUB`, and `WG` under `Graz` and `BrainHero` interfaces.
- **Auxiliary Public EEG Corpora**: `Zenodo 8089820` (`CC-BY-4.0`, motor imagery), `DepositOnce 5830.2` (`CC-BY-4.0`, cued word generation and resting baseline), and `PhysioNet eegmat 1.0.0` (`ODC-BY-1.0`, mental arithmetic and resting baseline).
- **Pretrained Public EEG Checkpoints (`HuggingFace`)**: `braindecode/eegpt-pretrained`, `braindecode/codebrain-pretrained`, `braindecode/cbramod-pretrained`, and `braindecode/mirepnet-pretrained` (`BSD-3-Clause` / `MIT`).
