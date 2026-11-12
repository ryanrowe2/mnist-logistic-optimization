# MNIST Logistic Regression & Numerical Optimization

**NumPy logistic regression, mathematical gradient verification, and reproducible first-order optimization.**  
**Author:** Ryan Rowe · University of California, San Diego · B.S. Mathematics–Computer Science, 2025  
**Provenance:** Reconstructed from UC San Diego MATH 173A coursework and the [July 18, 2024 report](docs/sources/2024-original-report.pdf).

The engineering question is not simply whether logistic regression classifies digits: **Can a first-order method be derived, implemented, verified, reproduced, and interpreted correctly?** This repository implements the objective and gradients in NumPy, then studies full-batch gradient descent and descent directions arising from different norm geometries. No scikit-learn estimator or optimizer is used.

## Evidence at a glance

| Finding | Status | Evidence |
|---|---|---|
| 4 vs 9: 96% accuracy on 1,000 evaluation samples | **2024 reported result; hosted rerun evidence linked separately** | [Primary report, pp. 13–16](docs/sources/2024-original-report.pdf) |
| 1,000 train, 1,000 evaluation, 10,000 updates, η = 0.0001 | **Recovered historical settings** | [Original experiment reconstruction](docs/original-experiment.md) |
| Analytic gradient, stable softplus, safe sigmoid | **Implemented and independently tested** | [`logistic.py`](src/mnist_optimization/logistic.py), [`test_gradient.py`](tests/test_gradient.py) |
| Distinct historical and official MNIST protocols | **Implemented and tested offline** | [`data.py`](src/mnist_optimization/data.py) |
| Three first-order descent rules | **Implemented and tested offline** | [`optimizers.py`](src/mnist_optimization/optimizers.py) |
| OpenML MNIST experimental reruns | **Not executed in this offline build** | [Validation status](docs/limitations.md) |
| Provenance and transactional artifact integrity | **Implemented and regression-tested** | [`provenance.py`](src/mnist_optimization/provenance.py), [`artifacts.py`](src/mnist_optimization/artifacts.py) |
| Train-only hyperparameter selection | **Implemented and tested offline** | [`tuning.py`](src/mnist_optimization/tuning.py) |
| Offline mathematical/integration suite | **294 offline tests passed; 86.06% combined package/script coverage; hosted certification tracked in release status** | [Release status](docs/release-status.md) |

**Provenance boundary:** A locally computed SHA-256 identifies bytes received; it does **not** certify upstream identity. No caller-issued token or source label can mark an input as independently verified canonical MNIST. The numerical report uses only reviewed, unverified source labels; arbitrary caller text is stored separately as `user_supplied_source_note_unverified` and never used in figure captions. The numerical output records the unverified arrays and any structurally reconstructed selection; the command-line program may separately print that it invoked the OpenML loader, which does not authenticate the dataset.

**Scientific integrity:** The 2024 result is a **historical reported result**, not a new benchmark. Synthetic fixtures validate execution but are **not** represented as MNIST classification observations. The repository intentionally ships no invented MNIST plots or replicated accuracies.

## Publication-package integrity and updated empirical metrics

The repository has a portable ZIP builder and an allowlist-enforcing verifier (v1.15.1):

```bash
python scripts/build_release.py --output ../mnist-logistic-optimization-release.zip
python scripts/verify_release.py ../mnist-logistic-optimization-release.zip
# Extract the ZIP, and run this from the extracted project directory:
sha256sum -c MANIFEST.sha256
```

All manifest paths are **relative to the extracted project root**, not to the build computer.
The release regression suite checks clean extraction, wheel and CLI execution, malicious self-consistent manifests, archive expansion limits, unexpected paths, and symlinks. The root `_SUCCESS.json` marker inventories **files and directories**; it is not an authentication signature for OpenML.
Public `run_prepared` always labels its data `user_supplied_prepared_unverified` and never certifies
historical or official source selection. Only the dataset-driven pipeline validates those selection protocols.

Newly executed held-out evaluations report 95% Wilson intervals for overall accuracy and class recall.
**The 2024 rounded 96% claim has no newly computed interval** because the original exact predictions
were not retained and real MNIST could not be obtained here.

For a platform-specific hash-locked reference environment, validate the complete wheelhouse and then perform a fresh isolated installation. Run on a network-connected
Python 3.13/Linux machine:

```bash
python -m pip download --only-binary=:all: -d wheelhouse \
  -r requirements-validation.txt 'ruff>=0.8,<1' wheel packaging
python scripts/create_platform_lock.py --wheelhouse wheelhouse \
  --output locks/reference-py313-linux-hashed.txt
python -m pip install --no-index --find-links wheelhouse --require-hashes \
  -r locks/reference-py313-linux-hashed.txt
# In a fresh target environment, import the locked NumPy, Matplotlib,
# scikit-learn, pytest, and Ruff distributions to verify installation.
```

This is a prepared **lock-generation and verification workflow**; no downloaded wheel hashes
have been claimed as locally verified. GitHub CI provides a manual generation job after publication.

For full real-MNIST verification (OpenML required), run:

```bash
python scripts/verify_real_mnist.py --output-root results/full_real_validation
```

This command executes both full historical and improved runs and saves validated evidence.
The archive contains no newly claimed real-MNIST results; see
[the latest remediation register](docs/release-status.md).

**Current verification (v1.15.1):** 294 offline tests passed with 1 network test deselected; 86.06% combined coverage on Python 3.12. Final exact-ZIP evidence is documented separately. The accompanying certification report records the exact extracted-release results. See [release status](docs/release-status.md) for evidence and external gates. This release additionally requires a clean, tracked Git source tree for commit attribution; makes train-only validation independent of label sign; rejects lossy extended-precision inputs; and prevents staging/manifest symlink overwrites. The preceding release separates caller-provided source notes from enumerated unverified source labels, rejects duplicate JSON keys in completion manifests, and saves training predictions for independent metric checks. Ruff lint and formatting, the hosted Python matrix, genuine OpenML smoke, and the Python 3.13/Linux hash-locked installation have passed. Full reproduction outputs and immutable run evidence are tracked in [release status](docs/release-status.md).

## Algorithms and mathematics

The binary labels are \(y_i \in \{-1,+1\}\), and the unregularized logistic objective is

\[
F(w)=\frac{1}{n}\sum_{i=1}^{n}\log\left(1+\exp(-y_i x_i^\top w)\right).
\]

The analytical gradient is

\[
\nabla F(w)=-\frac{1}{n}X^\top\left[y\odot\sigma(-y\odot Xw)\right].
\]

The implementation evaluates log-loss with `np.logaddexp` and the sigmoid term with a stable log-domain expression. It supports optional L2 regularization, with an unpenalized intercept in the improved protocol.

Three implemented update rules use different geometries:

| Optimizer | Direction \(d\), with \(g=\nabla F(w)\) | Update |
|---|---|---|
| Euclidean (`gd`) | \(g\) | \(w\leftarrow w-\eta g\) |
| \(L_\infty\) (`linf`) | \(\lVert g\rVert_1\operatorname{sign}(g)\) | \(w\leftarrow w-\eta d\) |
| \(L_1\)-geometry greedy coordinate (`coordinate`) | \(g_{j^*}e_{j^*}\), \(j^*=\arg\max_j\lvert g_j\rvert\) | \(w\leftarrow w-\eta d\) |

The latter two are the norm-steepest directions stated in the coursework assignment; **they correct discrepancies in the older submitted Python code**. They are *not* automatically directly comparable at arbitrary learning rates. See [mathematical derivations](docs/mathematical-foundations.md) and [optimization analysis](docs/optimization-analysis.md).

## Two evaluation protocols — never pooled

| Property | Historical reconstruction | Improved evaluation |
|---|---|---|
| Source | OpenML `mnist_784`, v1, ID 554 | Same, canonical MNIST ordering |
| Training | First 500 occurrences of each digit in all 70k rows | Seeded sample of 2,000 of each digit from official first 60k rows |
| Evaluation | Next 500 occurrences of each digit from all 70k rows | Seeded sample of 500 of each digit from official final 10k rows |
| Features | 784 raw 0–255 intensities | 784 fixed /255 features + intercept |
| Baseline optimizer | 10,000 GD updates, η=1e-4, loss recorded after each update, no regularization | 2,000 GD updates, η=0.3, λ=0.01 |
| Splits | Disjoint within the same class-order stream; **not** the official test split | Disjoint MNIST source partitions; test observations strictly held out |
| Seed | 2024 for *reconstructed* shuffle only; original seed unrecorded | 2024 for sample selection and shuffle |

The official protocol keeps the MNIST-designated 60,000/10,000 split. The fixed intensity division requires no learned preprocessing statistics. **Test data never informs parameter updates or hyperparameter selection.** The two baseline configurations are historical/demonstration choices; a separate, new [train-only validation utility](src/mnist_optimization/tuning.py) can select learning rates without using test labels. See [evaluation methodology](docs/evaluation-methodology.md).

## Installation and reproduction

Requires Python 3.10+; a network connection is needed on the first OpenML fetch unless scikit-learn has already cached dataset 554.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python -m pytest -q --cov=mnist_optimization --cov-report=term-missing
ruff check .
```

To execute the **historical 4 vs 9 reconstruction** on **real OpenML MNIST**:

```bash
python scripts/reproduce_original.py
# Or configure the same experiment explicitly:
mnist-optimize --protocol historical --digits 4 9 --optimizer gd \
  --step-size 0.0001 --iterations 10000 --seed 2024 \
  --output results/historical_4_vs_9
```

To execute the **separate official-split experiment**:

```bash
python scripts/run_experiments.py
# Equivalent CLI:
mnist-optimize --protocol official --digits 4 9 \
  --train-per-class 2000 --test-per-class 500 \
  --optimizer gd --step-size 0.3 --l2 0.01 --iterations 2000 \
  --seed 2024 --output results/official_4_vs_9
```

To choose method-specific learning rates **without tuning on the official test partition**, run:

```bash
python scripts/tune_optimizers.py --protocol official --iterations 500 \
  --gd-grid 0.05 0.1 0.3 --linf-grid 0.0001 0.0005 0.001 \
  --coordinate-grid 0.5 1 2 --output results/train_only_tuning
```

The training subset is stratified into selection-training/validation, the winning step per optimizer is selected using **validation loss only**, then each model is refitted on the entire training subset and evaluated once on the untouched test subset. The resulting `selection.json`, `train_validation_rows.npz`, validation-step plot, individual training curves, held-out metrics, and nested success manifests preserve the comparison procedure. This is a **new** extension, not historical coursework.

To experiment with the corrected alternative directions, specify `--optimizer linf` or `--optimizer coordinate` and deliberately inspect step sizes and loss trajectories. A single-split comparison overlay can be generated with explicit step sizes:

```bash
python scripts/compare_optimizers.py --gd-step 0.0001 --linf-step 0.00000001 \
  --coordinate-step 0.0001 --iterations 1000
```

These step sizes are illustrative coursework-derived values, **not tuned** or asserted to be fair across descent geometries. The script saves three full method runs and a machine-readable comparison/overlay. Observe and investigate trajectories instead of inferring a universal best optimizer. All methods use NumPy, not scikit-learn's model training routines. For the 0/1 historical baseline, use `--protocol historical --digits 0 1`.

Each successful run writes `metrics.json`, `numeric_results.npz` (weights, indices, objective observations, training/test true labels and predictions), a SHA-256 `_SUCCESS.json` inventory, and figures for convergence, confusion, real evaluation samples, potential mistakes, and 28×28 weights. The coordinate method additionally draws a coordinate-update heatmap. **Outputs are immutable:** choose a new/empty `--output` directory for each run. Failed runs are rolled back before publication. Output files are generated by the scripts, not included as precomputed MNIST claims.

To reproduce a checked-in **synthetic, two-dimensional algorithm demonstration**, requiring no MNIST download:

```bash
python scripts/generate_synthetic_validation.py
```

The synthetic diagram is explicitly named as a mathematical smoke demonstration, *not* an MNIST learning curve. You can run a trusted manually acquired local canonical-format array with `--local-npz path/to/mnist.npz`, with `X` shape `(70000,784)` and `y` length `70000`; for source integrity the CLI will **label it unverified**, even if it looks like MNIST.

## Repository navigation

- [`src/mnist_optimization/logistic.py`](src/mnist_optimization/logistic.py): objective, gradient, validation, predictions.
- [`src/mnist_optimization/data.py`](src/mnist_optimization/data.py): OpenML acquisition and leakage-controlled sample construction.
- [`src/mnist_optimization/optimizers.py`](src/mnist_optimization/optimizers.py): reproducible descent methods, loss histories, update counts.
- [`src/mnist_optimization/experiments.py`](src/mnist_optimization/experiments.py): CLI, provenance, metrics JSON, numeric artifacts.
- [`scripts/compare_optimizers.py`](scripts/compare_optimizers.py): controlled, three-method single-split runs and comparison figure.
- [`tests/`](tests): finite-difference, extreme-input, split integrity, optimizer, CLI, provenance-spoofing, artifact rollback, live-loader mocks, and train-only tuning tests.
- [`docs/original-experiment.md`](docs/original-experiment.md): actual 2024 methodology and historical result limitations.
- [`docs/limitations.md`](docs/limitations.md): remaining evidence gaps and scientific caveats.
- [`docs/release-status.md`](docs/release-status.md): maintained verification status and external release gates.

## What this demonstrates

**Software engineering:** A small installable Python package, input contracts, real CLI, tests, hosted CI, deterministic scientific JSON, transactional outputs, input hashes, and documented failure modes. **Data analysis:** Explicit dataset provenance, correct separation of evaluation observations, confusion matrices, per-class recall, and an honest treatment of performance uncertainty. **Applied mathematics:** Convexity, smoothness, finite differences, stability, and norm-dependent descent directions.

**Acknowledgment:** The 2024 academic work is the basis for this reconstruction. Later code organization, numerical safeguards, tests, and evaluation protocol are modern portfolio revisions, not represented as original 2024 submissions. Dataset rights remain with the original MNIST rights holders. [Attribution](ATTRIBUTION.md).

## Release verification and explicit limitations

The v1.15.1 hosted Python 3.10, 3.12 and 3.13 jobs each passed **294 offline tests** with combined coverage of **86.37%, 86.06% and 86.06%**, respectively. Ruff lint and formatting, release ZIP verification, distribution wheel installation and CLI checks passed. The genuine OpenML smoke test and Python 3.13/Linux wheelhouse hash-lock installation also passed. See [release status](docs/release-status.md) for the full reproduction measurements and immutable certification links.

The validated Python 3.13/Linux dependency hash lock is provided at [`locks/reference-py313-linux-hashed.txt`](locks/reference-py313-linux-hashed.txt). Its 27 acquired wheels were installed offline with required SHA-256 hashes and package-import checks. The earlier version snapshot remains at [`locks/reference-py313-linux.txt`](locks/reference-py313-linux.txt). Refer to the [release gates](docs/release-status.md) before publishing performance claims. Historical mode uses the source notebook's exact `np.sign` prediction ties (an ambiguous score 0 is counted as an error); official mode ties predict +1. See [evaluation methodology](docs/evaluation-methodology.md).


### Provenance scope (v1.9.0)

Public APIs processing user-provided arrays never claim to have invoked OpenML;
`selection.json` and `comparison.json` always record `not_observed_by_array_api`.
The two standalone CLI entry points emit a separate
`cli_only_acquisition_observation` in their **stdout** after the selected
loader returns. Capture stdout when retaining execution provenance. This records
a local execution route and does **not** verify canonical MNIST identity.

Output paths containing symlinks (including ancestors) or `..` traversal are
rejected before staging to prevent unintended write redirection.

## v1.9.0 engineering and provenance improvements

Source fingerprints use installation-independent relative member names; installed-wheel and source-checkout fingerprints have distinct scopes. Git revisions are accepted only for the MNIST checkout itself. Coordinate heatmaps show **effective parameter changes**, while `selection_counts` documents selected coordinates; zero gradients and steps rounded to no-op contribute no effective updates. See [release status](docs/release-status.md) for verification gates.
