# v1.14.0 — Coordinated numerical and release-safety verification

This version addresses the ten findings in the independent v1.13.0 whole-repository audit. The 2024 MNIST accuracy remains **historical reported evidence**. No newly downloaded dataset, current OpenML accuracy, hosted CI success, or third-party package lock is fabricated.

| Finding | Change | Regression evidence |
|---|---|---|
| A01 digit lists | Canonicalize valid `Configuration.digits` lists to immutable tuples | `test_a01_list_digit_pair_matches_canonical_split` |
| A02 overflowing mean | Normalize logistic loss terms before reduction; use compensated `math.fsum` | `test_a02_mean_of_representable_extreme_losses` |
| A03 gradient sum overflow | Fast BLAS with wider-precision fallback for representable extremes | `test_a03_extreme_representable_gradient` |
| A04 half-L2 overflow | Apply half-scaling before squared-norm accumulation | `test_a04_representable_regularization` |
| A05 finite cancellation | Compute pathological dot products in wider precision only when needed | `test_a05_finite_cancelling_products` |
| A06 output-link overwrite | Reject symlinked paths and publish the lock atomically | `test_a06_lock_output_symlink_rejected_without_touching_canary` |
| A07 external wheel link | Reject wheelhouse links and open regular wheel files without following symlinks | `test_a07_external_wheel_symlink_rejected` |
| A08 incomplete wheelhouse | Verify project/validation roots and active transitive requirements against supplied wheel versions | `test_a08_reject_missing_required_wheels` |
| A09 malformed wheels | Parse wheel metadata, supported tags, RECORD entries and hashes | `test_a09_reject_fake_wheel_even_with_expected_name` |
| A10 output-link overwrite | Guard synthetic JSON/PNG paths and publish through temporary files | `test_a10_synthetic_outputs_do_not_follow_symlinks` |

The normal numerical path remains vectorized NumPy. The rare extended-range fallback is not used for ordinary MNIST data. A mathematically unrepresentable float64 result continues to fail explicitly. The reference hash-lock generator now validates **structural completeness of a local wheelhouse**. A genuine reference lock is certified only **after** a clean `pip install --no-index --require-hashes` from that wheelhouse and real package import smoke checks; the isolated fake-wheel fixture establishes only the code path, not external package availability.

## Explicit outstanding certification

- Real OpenML MNIST historical and official-split runs are not verified in this sandbox.
- The complete Ruff `E,F,I,B,UP` ruleset has not run unless an executable is available; syntax and targeted static checks are not the same.
- Hosted GitHub Actions remains unexecuted until authorized publication.
- A real platform-specific, hash-locked dependency installation still requires complete third-party wheels.

## Verification commands

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m pytest -p pytest_cov -m 'not network' --cov=mnist_optimization --cov-report=term-missing
python scripts/build_release.py --output ../mnist-logistic-optimization-v1.14.0.zip
python scripts/verify_release.py ../mnist-logistic-optimization-v1.14.0.zip
# Run on a network-connected target with complete dependency wheels:
python scripts/create_platform_lock.py --wheelhouse wheelhouse --output locks/reference-local-hashed.txt
python -m pip install --no-index --find-links wheelhouse --require-hashes -r locks/reference-local-hashed.txt
```

The final verification counts, archive hash, and observed executable checks are recorded in the separate delivered verification report, rather than copied forward from an earlier release.
