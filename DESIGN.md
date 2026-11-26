# NeurIPS BCI Software & Signal-Processing Design Document

This document specifies the object-oriented software architecture, strict dependency layering, naming contracts, and causal signal-processing design of `neurips_bci`.

## 1. Domain Terms and Abbreviations

Every domain and engineering term used in this specification is defined here in plain words:

- **Electroencephalography (EEG)**: recording electrical brain potentials from electrodes placed on the scalp, measured in **microvolts (uV, millionths of a volt)** at a fixed **sampling rate in hertz (Hz, samples per second)**.
- **Brain-computer interface (BCI)**: software that decodes brain activity into discrete commands without muscle movement.
- **Motor imagery (MI)**, **mental subtraction (SUB)**, and **word generation (WG)**: the three mental tasks decoded in each 4-second (`41 x 480`) window. Motor imagery suppresses 8 to 30 Hz **sensorimotor rhythm (SMR)** power over contralateral motor cortex, a phenomenon called **event-related desynchronization (ERD)**.
- **Causal online streaming (`B = 1`)**: an evaluation contract in which test windows of a recording **session** arrive one at a time in stream order; predictions at window `t` may depend only on windows `0..t` of that session, and `reset_state()` clears all session memory at session boundaries.
- **Spatial covariance**: a symmetric `41 x 41` matrix measuring pairwise co-variation between electrodes inside a frequency band.
- **Euclidean alignment (EA)**: whitening each EEG window by multiplying by $R_t^{-1/2}$, the inverse square root of the causal session reference covariance matrix $R_t$, so cross-person skull and impedance scaling differences shrink.
- **Riemannian tangent space**: mapping symmetric positive-definite covariance matrices from their curved manifold onto a flat Euclidean vector space at a reference point so linear models can classify them.
- **Common spatial patterns (CSP)** and **linear discriminant analysis (LDA)**: supervised spatial filters maximizing between-class variance ratios paired with a linear Gaussian classifier.
- **Convolutional neural network (CNN)**: a neural network applying short temporal and spatial filter banks followed by variance pooling.
- **Batch normalization (BN)** and **test-time entropy minimization (Tent)**: normalizing internal layer activations using causal running statistics (`BN`) and updating scale/shift parameters online to reduce prediction entropy (`Tent`).
- **Self-training (ST)**: mid-session weight refitting (`t = 66, 126`) on earlier session windows labeled by a high-confidence multimodal teacher.
- **Foundation model**, **low-rank adaptation (LoRA)**, and **singular value decomposition (SVD)**: a large neural encoder pretrained on external EEG corpora, adapted via narrow rank-8 matrix factors (`LoRA`) or compressed via orthogonal variance factorization (`SVD`).
- **Sinkhorn-Knopp scaling**: iterative row-and-column normalization of class probabilities so per-window probabilities sum to 1 while session-cell class totals stay balanced.
- **Leave-one-subject-out (LOSO)** and **cross-session (XS)**: cross-validation splits testing unseen participants (`LOSO`, 10 folds) and later sessions of known participants (`XS4`, 4 folds; `XS1`, single-prior-session folds), combined in **`k05`** (`0.5 * mean(LOSO) + 0.5 * mean(XS4)`) across recorded order **`O0`** and five shuffled window orders **`OW5`**.
- **Abstract syntax tree (AST)** and **secure hash algorithm 256-bit (SHA-256)**: parsed Python code structure used for static rule verification (`AST`) and cryptographic file digests (`SHA-256`) used for archive integrity.

---

## 2. Eleven-Layer One-Way Object-Oriented Architecture

`neurips_bci` is organized into **11 strictly ordered package layers (`L0` through `L10`)**. Imports move downward only: a module in layer $L_i$ may import from `neurips_bci` layers $L_0 \dots L_i$ and never from $L_j$ with $j > i$.

```text
  +-------------------------------------------------------------------------+
  | L10  workflow/       Workflow -> StageWorkflow, CliWorkflow             |
  |                      Orchestrates download, extract, train, package,    |
  |                      verify, evaluate, reproduce, and CLI entrypoints.  |
  +------------------------------------+------------------------------------+
                                       | imports downward only
                                       v
  +-------------------------------------------------------------------------+
  | L9   evaluator/      Evaluator -> MetricEvaluator, BenchmarkEvaluator   |
  |                      Cell-mean balanced accuracy, k05, 10-cluster       |
  |                      jackknife SE, 1,024 sign-flip p, 16-fold B=1 CV,   |
  |                      and 6-gate archive contract survival verifier.     |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L8   decoder/        Decoder -> EnsembleDecoder                         |
  |                      Top-level causal B=1 BCI decoder integrating       |
  |                      SessionPaceGate, ArtifactGuard, PseudoChrono EA,   |
  |                      organs, Sinkhorn, G1 reliability gate, reset_state.|
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L7   organ/          Organ -> CnnOrgan, FoundationOrgan,                |
  |                               RiemannianOrgan, MultimodalOrgan          |
  |                      Self-contained decoding organs: CNN40 dual-block,  |
  |                      EEGPT-19, TSR4, RSNSE, ANCHOR-MM, CodeBrain+LoRA8, |
  |                      CBraMod, MIRepNet, S3D Fisher anchored ridge.      |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L6   backbone/       Backbone -> ConvolutionalBackbone,                 |
  |                                  FoundationBackbone                     |
  |                      PyTorch neural architectures: ShallowConvNet,      |
  |                      FBCNet, FBModNet, EEGPT, CodeBrain, CBraMod,       |
  |                      MIRepNet, grouped & CUDA-graph wrappers.           |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L5   adaptation/     Adapter -> SignalAdapter, ProbabilityAdapter,      |
  |                                 GateAdapter                             |
  |                      Causal session state machines: ArtifactGuard,      |
  |                      PseudoChrono EA/BN, running/Kalman/Sinkhorn        |
  |                      balancers, G1 & confidence gates, PaceGuard.       |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L4   acquisition/    Acquirer -> CorpusAcquirer, CheckpointAcquirer,    |
  |                                  SplitAcquirer                          |
  |                      Public dataset & checkpoint downloaders and        |
  |                      41-channel 120 Hz window extractors (NEMAR,        |
  |                      Zenodo, DepositOnce, PhysioNet, HuggingFace).      |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L3   container/      Container -> StreamContainer                       |
  |                      Typed immutable/safe-npz EEG data structures:      |
  |                      WindowedCorpusStreamContainer,                     |
  |                      SessionWindowStreamContainer,                      |
  |                      ValidationFoldStreamContainer.                     |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L2   configuration/  Configuration -> SectionConfiguration,             |
  |                                       RunConfiguration                  |
  |                      ExperimentRunConfiguration is the SOLE JSON I/O    |
  |                      boundary and BLAKE2b deterministic seed source.    |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L1   operator/       Operator -> MatrixOperator, SignalOperator         |
  |                      Pure stateless math: shrinkage covariances,        |
  |                      matrix inverse square roots, tangent projections,  |
  |                      Butterworth/FIR filter banks, spherical splines.   |
  +------------------------------------+------------------------------------+
                                       |
                                       v
  +-------------------------------------------------------------------------+
  | L0   validator/      Validator -> TypeValidator                         |
  |                      Construction-time ParameterTypeValidator and       |
  |                      AST/SHA-256 ArchiveSafetyTypeValidator.            |
  +-------------------------------------------------------------------------+
```

### 2.1 Family, File, and Naming Invariants

Every Python file inside `neurips_bci/` obeys five machine-checked rules enforced by `tests/test_codebase_conventions.py`:

1. **One Class Per File**: Every `.py` module (excluding `__init__.py`) defines exactly one top-level class, and the filename matches `<ClassName>.py`.
2. **Family Hierarchy & Plural Sub-Family Folders**: Each layer folder (`validator/`, `operator/`, `configuration/`, `container/`, `acquisition/`, `adaptation/`, `backbone/`, `organ/`, `decoder/`, `evaluator/`, `workflow/`) places its abstract family root class (`Validator`, `Operator`, `Configuration`, `Container`, `Acquirer`, `Adapter`, `Backbone`, `Organ`, `Decoder`, `Evaluator`, `Workflow`) directly inside the family folder. Sub-families live in plural subfolders (for example `evaluator/metric_evaluators/` and `evaluator/benchmark_evaluators/`), each containing an abstract sub-family parent (`MetricEvaluator`, `BenchmarkEvaluator`) and concrete variant classes whose names end with the sub-family and family words (`CellBalancedAccuracyMetricEvaluator`, `CrossValidationBenchmarkEvaluator`, `ArchiveContractBenchmarkEvaluator`).
3. **Explicit Domain Identifiers**: Generic identifiers (`name`, `data`, `value`, `result`) are forbidden across all class names, method names, parameter names, local variables, and attributes. Every identifier states what it represents and where it comes from (`config_identifier`, `ordered_windows`, `fold_deltas`, `inspection_report`).
4. **Code Says How, Documents Say Why**: Python source files contain zero academic paper citations; rationales, empirical trade-offs, and dataset attributions live in `README.md`, `DESIGN.md`, and `NOTICE.md`.
5. **Single Configuration Boundary & Single Master Seed**: Only `ExperimentRunConfiguration` reads or writes **JavaScript Object Notation (JSON)** configuration files. All downstream classes accept plain typed constructor arguments verified at build time by `ParameterTypeValidator`. Every workflow run saves `resolved_run_configuration.json` in `output_run_directory` and derives deterministic component seeds from `master_random_seed` via `derive_component_seed(component_role_key, member_index)`.

---

## 3. H10 Causal Signal-Flow Specification

The schematic below traces a single streamed window $x_t \in \mathbb{R}^{1 \times 41 \times 480}$ through the `L5..L8` runtime components and shows how `Submission 7` (`h10_r45_submission7`) refines `Submission 6` (`h10_r35_submission6`) without changing a single Tier-0 classification decision:

```text
  x_t (1, 41, 480)
   |
   +--> [G] SessionPaceGateAdapter --------> active_tier tau_t in {0, 1, 2, 3}
   |        r35: b=0.205 s/w, global T2=3700s, T3=4100s, T1 drops ST
   |        r45: b=0.290 s/w, global T1=3550s, T2=4300s, T3=4420s, T1 keeps ST
   |
   +--> [P] Mean-Center & NaN/Inf Zeroing -> x_clean (41, 480)
   |
   +--> [A] CausalArtifactSignalAdapter ---> x_guard (41, 480), clean_flag ok_t in {0, 1}
   |        z_amp <= 8.0, z_logdet <= 8.0, soft-clip > 32.0 robust scales
   |
   +--> [E] SpectralSignalOperator + PseudoChronoSignalAdapter
   |        6 bands + broadband covariances {C_t^(b)}; 6-D drift kernel K_{t,s} (s < t)
   |        whitened signal w_t = R_t^(-1/2) x_guard
   |
   +--> [C] DualBlockCnnEnsembleCnnOrgan --> u_cnn40 (3,)
   |        Block A (offset 200) + Block B (offset 1200): 160 nets
   |        (Shallow25, FBCNet, distilled FBModNet, distilled Shallow49)
   |        + PseudoChrono BN + Tent (t < 130) + ST & AG3 at t in {66, 126}
   |        + Tangent LR + Session CSP/LDA + Participant Specialists
   |
   +--> [F] Multimodal Parallel Organs (gated by active_tier tau_t):
   |        - EEGPT-19 (tau_t <= 2): u_lat52, u_rms56 (Cholesky refit at 66, 126)
   |        - TSR4 (tau_t <= 2): u_tsr4
   |        - RSNSE (tau_t <= 1): u_rsnse
   |        - ANCHOR-MM (tau_t <= 1; GPU float64 in r45): residuals (r_dm, r_pr)
   |        - CBraMod (tau_t <= 1): u_cb (w = 0.10)
   |        - CodeBrain + Rank-8 LoRA (tau_t == 0): u_cd (w = 0.15)
   |        - MIRepNet (tau_t == 0): u_mr (w = 0.10)
   |        - S3D 16-D Fisher AncRidge (tau_t == 0): u_s3d (w = 0.12)
   |
   `--> [M] H10MultimodalEnsembleDecoder Fusion:
            TrajA (0.40) + TrajB (0.60) -> + ANCHOR-MM (emits ST teacher at 66, 126)
            -> + 0.06 Shallow_STOFF -> PseudoChrono Sinkhorn (tau=1.3, 4 iters)
            -> + G1-gated (CodeBrain, CBraMod, MIRepNet) -> + 0.12 S3D -> argmax_c p_{t,c}
```

### 3.1 Six-Gate Archive Survival Contract (`ArchiveContractBenchmarkEvaluator`)

Before any packaged folder or `.zip` archive is certified for competition upload, `ArchiveContractBenchmarkEvaluator` executes six automated verification gates:

1. **Static Archive Safety (`Gate 1`)**: `ArchiveSafetyTypeValidator` checks flat directory layout (zero nested paths), single `submission.py`, zero `.pkl`/`.pickle`/`.joblib` files, zero forbidden network or `pickle` AST nodes (`allow_pickle=True`, `weights_only=False`), and exact `MANIFEST_SHA256.json` digest parity.
2. **Read-Only + Network-Blocked Load (`Gate 2`)**: Copies the bundle to a read-only (`chmod 0555`) temporary directory, blocks `socket.socket` creation to forbid runtime network calls, dynamically imports `submission.py`, instantiates `Solver().load_model(meta)`, and verifies `B = 1` inference.
3. **Strict Prefix-Sabotage Causality (`Gate 3`)**: Streams a multi-window sequence crossing a self-training checkpoint with `R31_FORCE_DET=1`, then corrupts all windows strictly after `t_cut` with large Gaussian noise and asserts `max_abs_diff == 0.0` on all outputs `0..t_cut`.
4. **Session-Order & `reset_state()` Invariance (`Gate 4`)**: Streams `[Session A, Session B]` and `[Session B, Session A]` using per-session `copy.deepcopy` + `reset_state()` (as well as sequential `reset_state()` on one instance) and asserts zero cross-session state leakage (`max_abs_diff == 0.0`).
5. **Hostile EEG Input Survival (`Gate 5`)**: Feeds pathological `(1, 41, 480)` streams (`NaN`, `+Inf`, `-Inf`, all-zero flatline, single dead channel, `5,000 uV` spike, and short 5-window session) and asserts finite `{0, 1, 2}` predictions without exceptions.
6. **Reference Parity (`Gate 6`, optional)**: Compares predictions on identical windows against a reference archive or `Decoder`, reporting decision-flip count (`0` required at Tier 0 between `r35` and `r45`) and maximum probability deviation.

---

## 4. Design History

### 2026-11-23 — Clean-Room 11-Layer Object-Oriented Decomposition of PROTEUS H10

- **Decision**: Decompose the monolithic 16,008-line research `submission.py` and multi-script Round 19–45 training harness into the 11-layer one-way `neurips_bci` package (`Validator` -> `Operator` -> `Configuration` -> `Container` -> `Acquirer` -> `Adapter` -> `Backbone` -> `Organ` -> `Decoder` -> `Evaluator` -> `Workflow`), with one class per file, plural sub-family directories, `ExperimentRunConfiguration` as the sole JSON configuration I/O boundary, and `SubmissionPackagingStageWorkflow` generating a flat standalone `submission.py` bundle verified by `ArchiveContractBenchmarkEvaluator`.
- **Reason**: Competition Codabench workers require a flat archive containing a single `submission.py` and weight `.pt`/`.npz` files, which caused the research codebase to accumulate in one large file. Separating stateless mathematical operators (`L1`), typed configuration (`L2`), stream containers (`L3`), data acquirers (`L4`), causal state adapters (`L5`), neural backbones (`L6`), modular organs (`L7`), top-level streaming decoders (`L8`), statistical/survival evaluators (`L9`), and CLI workflows (`L10`) eliminates duplication, makes every organ independently testable on synthetic CPU fixtures, and preserves exact compatibility with both `h10_r35_submission6` (`0.823` banked sealed) and `h10_r45_submission7` (`~0.830` projected sealed, `-27%` latency).
- **Structure Diagram**:

```text
  configs/*.json ---> [L2 ExperimentRunConfiguration]
                              |
                              v
  [L10 CommandLineInterfaceWorkflow / EndToEndReproductionStageWorkflow]
     |-- [L4 Acquirers] --------> [L3 WindowedCorpusStreamContainer]
     |-- [L6 Backbones + L5 Adapters + L1 Operators] -> [L7 Organs]
     |-- [L8 H10MultimodalEnsembleDecoder] ---------> packaged flat .zip
     `-- [L9 ArchiveContractBenchmarkEvaluator & CrossValidationBenchmarkEvaluator]
```

### 2026-11-24 — Package Rename to `neurips_bci` and `MirepnetFoundationBackbone` Submodule Fix

- **Decision**: Renamed the top-level Python package, command-line entrypoint, and Conda environment to `neurips_bci` / `neurips-bci`, updated `SubspaceMatrixOperator.PROTEUS_41_CHANNELS` to match the exact 41-electrode BIDS channel order (`AF4`, `AF3`, `FCz`), and replaced `nn.ModuleDict` inside `MirepnetFoundationBackbone` attention projections with a plain `nn.Module` container registered via `add_module`.
- **Reason**: `neurips_bci` matches the public GitHub repository name directly without internal competition codenames, the exact BIDS electrode tuple aligns spherical-spline interpolation (`M_45_41`) with `nm000290`, and plain `nn.Module` avoids colliding with `nn.ModuleDict.keys()` / `nn.ModuleDict.values()` while keeping exact `state_dict` key parity (`transformer.{i}.0.fn.1.{keys,queries,values,projection}.*`) with `mirepnet_fp16.npz`.
