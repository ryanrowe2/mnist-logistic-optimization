# Complete v1.12.0 audit remediation — v1.13.0

**Source of findings:** Comprehensive independent v1.12.0 audit, November 12, 2026.
**Validation scope:** Entire source package, public CLIs, release builder/verifier,
Python compatibility, and regression/integration workflows.

| Audit ID | Finding | Resolution and regression evidence |
|---|---|---|
| H01 | ZIP temporary-path symlink overwrite | Exclusive unpredictable `mkstemp` staging, `O_NOFOLLOW` source reads, and atomic ZIP publication; canary preserved |
| H02 | Manifest symlink overwrite | Guard all ancestors and destination; exclusive manifest staging and atomic replacement; malicious manifest link rejected |
| H03 | Ruff B905 zip calls | Explicit `strict=False` for intentionally truncated plots and `strict=True` for equality tests; AST regression |
| M01 | Git SHA for untracked code | Require tracked package code, scripts, and project metadata before recording revision |
| M02 | Git SHA for dirty code | Omit source revision and label `git_revision_state=dirty_source`; source hash still identifies actual bytes |
| M03 | Tuned validation changes on label reversal | Single label-neutral seeded row permutation; validate exact same rows under `y → -y` |
| M04 | Lossy wider-than-float64 input | Reject any extended-precision feature value not exactly representable in float64 |
| M05 | Padding-dependent extended-float hashes | Canonicalize exactly representable extended floats to float64 before hashing; reject lossy input |
| M06 | Mutable RNG passed as seed | Enforce nonnegative built-in Python integer seeds in public splitter |
| L01 | List digit pair survives preparation | Canonicalize pair to a two-integer tuple; validate both protocol variants |
| L02 | Complex class labels admitted | Enforce real numeric binary label kind and finiteness |
| L03 | Boolean seed admitted | Reject bool, NumPy bool, and non-Python integer seeds |
| L04 | Caller-reported NPZ transport | Array source labels are unverified arrays; CLI observation retains actual transport separately |
| L05 | Fetch before occupied output preflight | Validate destination before invoking OpenML loader |
| L06 | Outdated README test counts | Update headline and release-status evidence for v1.13.0 |

## Engineering validation

- All 22 added adversarial tests pass, with multiple deliberately malicious file probes.
- Complete 267-test offline suite passes across four completed, non-overlapping groups.
- Source coverage: 93% (900 executable statements, 66 missed).
- The release builder must pass a separately extracted SHA-256/allowlist verifier,
  independent deterministic rebuild, and wheel and installed-CLI smoke tests.
- The original July 2024 report remains unmodified, historically attributed.

## Unexecuted external gates

Actual OpenML dataset 554 acquisition and both real-MNIST experiments; complete
Ruff execution; GitHub Actions hosted matrix; acquisition and hash-locked
installation of third-party dependency wheels. DNS/PyPI access is unavailable
in the current execution container, and GitHub publication has not been authorized.

Do not cite synthetic fixtures as real-MNIST accuracy, or claim these external
checks have passed. A successful SHA-256 inventory detects accidental changes;
it is not cryptographic authentication of dataset origin or of a third-party
release signer.
