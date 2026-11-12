# v1.15.0 — Consolidated remediation and release verification

## Scope

This release addresses N01–N10 from the v1.14.0 independent audit.
It includes selective exact binary64 product accumulation for cancellation,
wheel metadata validation and dependency closure with activated extras,
canonical ZIP paths and resource budgets, reference build dependencies,
and script-level coverage instrumentation.

## Evidence policy

A passing offline test on controlled fixtures is not evidence of MNIST
accuracy or a complete third-party dependency installation. Working-copy verification: 292 passed, 0 failed, 1 live-data test
deselected. Coverage includes 882/953 package statements (92.55%) and
493/656 executable-script statements (75.15%), combined 1375/1609
(85.46%). Final archive hashes and independent checks are recorded in the
accompanying release-verification report. Local Ruff execution
and GitHub Actions are not claimed unless their tools actually run.

## Verification commands

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p pytest_cov -m 'not network' \
 --cov=mnist_optimization --cov=scripts --cov-report=term-missing
python scripts/build_release.py --output ../mnist-logistic-optimization-v1.15.0.zip
python scripts/verify_release.py ../mnist-logistic-optimization-v1.15.0.zip
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/mnist-wheels
```
