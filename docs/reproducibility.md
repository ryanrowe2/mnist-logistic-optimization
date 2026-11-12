# Reproducibility and release verification

The maintained [release status](release-status.md) gives current verification evidence. See [engineering history](engineering-history.md) for reconstructed provenance.

## Local setup

From the extracted archive root on Python 3.10+ with a suitable package index:

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -p pytest_cov -m 'not network' --cov=mnist_optimization --cov-report=term-missing
ruff check .
python scripts/build_release.py --output ../mnist-release.zip
python scripts/verify_release.py ../mnist-release.zip
python -m pip wheel . --no-deps -w dist
```

The release builder includes only reviewed files specified in `release-files.json`, writes a relative SHA-256 manifest, and builds the ZIP deterministically. The independent verifier checks its contents, hashes, file types, path safety and expansion bounds. Outputs are committed atomically and cannot overwrite prior nonempty results. A completed result inventories all files and directories. Do not run tests *before* verifying a strict extracted directory: tests can create cache files that the verifier correctly rejects.

## Real MNIST (network or existing OpenML cache required)

```bash
python scripts/verify_real_mnist.py --output-root results/full_real_validation
```

This executes the historical reconstructed 4/9 protocol and a **separately labeled** official-split experiment. Inspect exact metric counts, confusion matrices, loss curves, full array/sample hashes, code fingerprint, and numerical outputs. Document any discrepancy with the historical rounded result without tuning to match it. Comparison/tuning array APIs record `acquisition_observation=not_observed_by_array_api`. Their CLIs emit a distinct `cli_only_acquisition_observation` to stdout after the corresponding local loader returns. Capture stdout if needed; verified upstream dataset identity remains **false**.

## Hash-locked reference environment

`locks/reference-py313-linux.txt` is a version snapshot, **not** a wheel hash lock. On a network-enabled Python 3.13/Linux environment:

```bash
python -m pip download --only-binary=:all: -d wheelhouse -r requirements-validation.txt 'ruff>=0.8,<1' wheel packaging
python scripts/create_platform_lock.py --wheelhouse wheelhouse --output locks/reference-py313-linux-hashed.txt
python -m pip install --no-index --find-links wheelhouse --require-hashes --target /tmp/locked-install -r locks/reference-py313-linux-hashed.txt
```

The optional GitHub Actions dispatch can execute these jobs **only after approval to publish**. No hosted CI, full real MNIST or independently installed hash-lock result is claimed unless evidence is available.

## Determinism and limits

Synthetic JSON and figure generation are checked by the offline suite. Fixed seeds constrain sample selection, including identity-keyed official train/test random streams and pair-order-invariant shuffling. Floating-point results may still vary slightly across hardware, BLAS, or NumPy versions. The recovered original report used a global unseeded shuffle, so its exact original permutation is unknown.
