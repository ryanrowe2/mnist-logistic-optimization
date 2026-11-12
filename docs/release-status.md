# Current release — v1.15.1

Targeted hosted-certification repair of the preserved v1.15.0 publication
(commit `f6b22f0bb40f8b2c10824c9c654644185a215270`).

- Ruff lint and canonical formatting pass with Ruff 0.17.5; both are enforced in CI.
- Local Python 3.12 regression run: **294 passed, one network test deselected,
  86.06% combined package/script coverage**, above the 80% threshold.
- MNIST/development extras and validation requirements include pandas.
- The OpenML loader explicitly selects scikit-learn's bundled liac-arff parser.
- Wheel validation accepts canonical empty directory records while preserving
  traversal, symlink, tag, RECORD, dependency and streaming-size checks.
- CI retains test reports, coverage, wheels, release ZIPs, genuine experiment
  outputs and the acquired wheelhouse with its hash lock for 30 days.

Hosted Python 3.10/3.12/3.13 jobs each passed 294 tests, with 86.37%/86.06%/86.06%
combined coverage. Ruff lint/formatting, release ZIP build/verification,
distribution wheels and installed CLIs passed. Genuine OpenML smoke passed;
the authentic Python 3.13/Linux wheelhouse passed validation, hash-locked offline
installation and package imports. Its 27 wheels match the SHA-256 hashes in
[`reference-py313-linux-hashed.txt`](../locks/reference-py313-linux-hashed.txt).

[Full certification workflow and experiment artifacts](https://github.com/ryanrowe2/mnist-logistic-optimization/actions/runs/43321505201)
record the 10,000-update historical and 2,000-update official-split executions.
The generated `full_real_mnist_summary.json`, numeric predictions, metrics,
figures and completion inventories are the measurement record. The delivered
closure report records their independently checked values and final run outcome.
[PR #1](https://github.com/ryanrowe2/mnist-logistic-optimization/pull/1) preserves
the complete targeted repair and final protected CI evidence. The original 2024 rounded 96% result remains historical evidence.

---

# Historical release — v1.15.0

This release addresses the ten findings in the comprehensive v1.14.0
whole-repository audit, with adversarial numerical and wheel-lock tests.
The numerical path detects finite cancellation as well as overflow.
The wheelhouse verifier checks extras, conflicting variants, internal wheel tags,
canonical member paths, bounded streaming validation and build requirements.
Python sources have targeted import-order corrections; the actual Ruff binary
is required for complete static-lint certification.

**Local working-copy validation:** 292 passed, 0 failed, one network test
deselected; 1,609 measured statements and 234 uncovered (85.46% overall).
Package modules: 882/953 statements (92.55%). Executable scripts:
493/656 statements (75.15%). A separate exact-archive verification is
required before release certification. See the accompanying report.

**External gates:** Real OpenML MNIST experiments, execution of the full Ruff
ruleset, hosted GitHub Actions, and installation from a real hash-locked
third-party wheelhouse remain unverified until independently executed.
The historical 96% MNIST accuracy is not presented as a new result.

[Detailed v1.15.0 verification](engineering-verification-v1.15.md).

---

# Historical release — v1.14.0

The latest local release repairs ten adversarial findings in extreme arithmetic and
hash-lock / generated-artifact filesystem safety. The original 2024 result is
historical evidence, not a newly executed accuracy measurement. Real MNIST,
full Ruff, hosted GitHub Actions and a complete platform-specific wheelhouse
installation remain unverified until those checks run successfully.

**Current local test evidence:** 279 passed, 0 failed, one network test
deselected, 92% source coverage (946 statements, 74 uncovered); see the
v1.14.0 verification report delivered with the ZIP. The 2024 accuracy
is historical and the observed local tests use offline fixtures.

[Detailed v1.14.0 correction inventory](engineering-verification-v1.14.md).

# Historical release status — v1.13.0

**Maintained status.** Source code, tests and mathematics are derived from the reconstructed 2024 report and separately labeled modern improvements. Previous audit versions are summarized in [engineering-history.md](engineering-history.md) rather than published as obsolete current assertions.

## Comprehensive v1.13.0 remediation (November 12, 2026)

This release closes the 15 verified findings from the independent v1.12.0
whole-repository audit, with 22 new adversarial regression tests. Major fixes:

- Safe, exclusive ZIP staging and atomic publication; refuse manifest/output
  symlinks, including preexisting `.incomplete` links, without overwriting
  files outside the project.
- Explicit `zip(..., strict=...)` to address known Ruff B905 violations.
  **The complete Ruff ruleset has not been executed in this offline runtime.**
- Git revisions require tracked and clean source modules, scripts, and
  `pyproject.toml`; record `git_revision_state` for dirty or untracked code.
- Class-sign-invariant train-only validation: row-keyed seeded random
  priorities preserve exact row membership under class-label reversal.
- Strict real binary labels, Python integer seeds, and immutable digit pairs.
- Reject lossily converted extended-precision feature values; normalize
  hashable extended floating values to canonical float64, rejecting values
  that would lose precision or depend on padding bytes.
- Caller-provided array source labels cannot assert an NPZ transport route;
  CLI-only acquisition observations remain separate and unverified.
- Reject occupied full-MNIST output destinations before network acquisition.
- Corrected the README headline test count and version metadata.

**Verified offline:** 267 passed, 0 failed, 1 network-dependent test
excluded; source-module coverage 93% (900 statements, 66 missed). Evidence
includes deliberate canary-based file overwrite attacks, clean Git checkouts,
untracked and dirty repositories, and the complete existing regression suite.

**Still unverified:** real OpenML MNIST historical/official runs, full Ruff,
hosted GitHub Actions, and a hash-locked wheelhouse installation. These are
release gates, not claimed successful results.

The detailed 15-finding register and reproduction tests are in
[v1.13.0 comprehensive remediation](engineering-verification-v1.13.md).

## Historical corrections in v1.12.0

1. Source-checkout detection parses `pyproject.toml` as TOML and requires
   `[project].name = "mnist-logistic-optimization"` before including scripts
   in the source hash or attributing the Git revision to the MNIST project.
   Python 3.10 uses the conditionally declared `tomli` dependency.
2. Official sampling uses digit-identity-keyed train/evaluation RNG streams
   and unordered-pair shuffle streams. Reversing `(4, 9)` to `(9, 4)`
   preserves selected images and ordering, while reversing every label.
   This changes the official-mode selections relative to earlier releases;
   it does not change the original historical sample-slicing protocol.
3. Public prepared-split validation rejects non-one-dimensional source-index
   arrays before optimizer execution or artifact creation.
4. Added adversarial regressions for foreign Git projects, pair reversal,
   stable sampling under test-set-size changes, and malformed indices.

**v1.12.0 local verification:** 245 offline tests passed across five
controlled, non-overlapping groups with no failures and one live-data
test deselected. Source-module coverage: **93%** (874 statements, 58
uncovered). Python 3.10 syntax parsing and configured 120-character Python
line-length checks passed. The final ZIP receives separate clean-extraction,
wheel-installation, CLI, and reproducible-build checks documented in the
accompanying certification report. Real MNIST, full Ruff, hosted CI, and
wheel-hash verification still require external execution.

## Corrections in v1.11.0

1. **Source-label integrity:** saved `source` fields and experiment figure captions
   use only reviewed, explicitly unverified categories. Free-text caller descriptions
   are retained in `user_supplied_source_note_unverified` and cannot impersonate
   an authenticated OpenML source in authoritative report fields.
2. **Unambiguous result completion:** `_SUCCESS.json` parsing refuses duplicate
   object keys at any depth and rejects unexpected top-level schema fields and
   noninteger schema versions. SHA-256 verification and full file inventories
   are preserved.
3. **Independently auditable training metrics:** numeric result archives now include
   `y_train` and `y_pred_train`, enabling recalculation of reported training
   accuracy and confusion counts without reaccessing the source images.
4. **Static lint remediation:** deprecated `typing.Iterator` replaced by
   `collections.abc.Iterator`; `zipfile` imports reordered. The complete Ruff
   ruleset still requires an environment with Ruff installed.
5. **Regression tests:** 15 targeted tests cover adversarial caller notes,
   duplicate completion keys (including nested maps), schema validation, and
   both train/test prediction evidence.

## Verification status for v1.11.0

**Updated checkout verification:** 230 offline tests passed, zero failures, one
live-data test deselected, across four completed non-overlapping test groups.
Package statement coverage: **94%** (855 statements, 52 uncovered). All Python
modules and tests parse under Python 3.10 syntax; all Python lines meet the
120-character configured limit. Exact-archive extraction, wheel installation
and build reproducibility are separately checked before certification.

Genuine OpenML, the full Ruff ruleset, and hosted GitHub CI remain separate
gates.

## Corrections in v1.10.0

1. **NumPy integer configuration:** exact-value-validated `np.integer` step sizes and L2 strengths are normalized to JSON-safe built-in floats, including signed and unsigned types. Nonrepresentable inputs fail before experiment execution.
2. **Provenance and protocol evidence:** the private experiment writer rejects reserved OpenML source labels, arbitrary digests, and caller-provided selection-validated flags. Full-array protocol claims are calculated only by validating the complete supplied arrays and independently reconstructing and comparing the exact split; acquisition provenance and canonical MNIST identity are never certified by these arrays. OpenML-loader calls may be reported by their CLI, not the numerical writer.
3. **Wilson interval API:** finite NumPy float16/32/64 confidence values are normalized and range-checked, and output remains JSON-serializable.
4. **Offline regression coverage:** new tests exercise signed/unsigned NumPy settings, forged internal metadata, split/full-data disagreement, genuine structural selection evidence, and Wilson NumPy scalars.

## Corrections in v1.9.0

1. **Comparable source fingerprints:** installed-wheel code is hashed with
   stable package-relative member names, not directory names of the installation
   environment. A checkout hashes package code, project entry points and
   pyproject metadata, with the scope clearly distinguished from a wheel digest.
2. **Accurate Git attribution:** a revision is recorded only when Git identifies
   the MNIST source checkout itself as its repository top level. Unrelated
   containing repositories are not credited as the project's revision.
3. **Truthful coordinate evidence:** zero gradients select no coordinate;
   `selection_counts` counts nonzero-gradient coordinate choices, and
   `update_counts` counts only representable changes to model weights.
   The coordinate heatmap represents effective updates, not selections or
   feature importance; both totals are saved in the experiment outputs.
4. **Regression coverage:** new tests check relocatable wheel-style hashes,
   foreign and owned Git repositories, stationary gradients, and rounded no-op
   updates. This is local engineering verification, not a completed MNIST run.

## Corrections in v1.8.0

1. **Output path safety:** transactional output staging rejects symlinks in the
   requested output root and any existing ancestor *before* path resolution or
   allocation. This matches completed-result verification.
2. **Acquisition observation:** public tuning and comparison APIs always record
   `not_observed_by_array_api`. CLI entry points print their own locally observed
   loader or NPZ route as `cli_only_acquisition_observation` to stdout **after**
   invoking that route. The observation does not authenticate upstream identity;
   redirects or retained CLI output should be treated as operator-side evidence.
3. **Official sampling documentation:** six independent seeded streams are fully
   specified in the evaluation methodology.

## Earlier corrections (v1.7.0)

1. **Official split:** six independent seeded streams: one training and one testing selection per class plus separate train/test shuffle streams. Test-set sample counts no longer change the training membership or its order. Historical slicing and seeded shuffling remain unchanged.
2. **Configuration serialization:** valid NumPy float16/float32/float64 scalar step sizes and regularization strengths become Python floats. Candidate grids and train-only validation fractions are normalized before JSON output. Invalid scalars still fail validation.
3. **Result integrity:** file and directory inventories reject symlinked ancestors as well as internal symlinks.
4. **Acquisition observation:** optimizer comparison and tuning scripts log that the CLI invoked OpenML or loaded a local NPZ. This observation is metadata, not authentication; `verified_openml`, `provenance_verified`, and canonical dataset identity remain false.
5. **Documentation:** replaced ten older audit logs with one concise historical provenance summary; the original 2024 PDF remains intact.

## Verification status

**Previously observed for v1.10.0 (historical):** 215 offline tests passed,
zero failed, and one network-dependent test deselected, across three completed
non-overlapping test groups with coverage instrumentation. Combined statement
coverage is **94% rounded** (838 statements, 52 uncovered). All 37 Python files
parse under Python 3.10 syntax rules. The exact release ZIP receives separate
clean-extraction, installation, and rerun verification before certification.
This is numerical and integration test evidence, **not** a completed OpenML run.


**Previously observed for v1.9.0:** 198 offline tests passed, zero failures, one
network-only test deselected across four independently completed coverage-instrumented
processes on Linux/Python 3.13.5. Combined package coverage: 822 statements, 52
uncovered, **94% rounded**. The earlier monolithic process timed out without a
reported assertion failure; the completed group runs, not that timeout, support
these numbers. The exact packaged archive must be independently verified after
building; see the separately delivered v1.9.0 certification report. Local tests
verify fixed scientific reporting and mathematical behavior on controlled fixtures;
**real MNIST was not executed**.


**Previously observed for v1.8.0 local checkout:** 190 offline tests passed, 0 failed, 1 network-dependent test deselected in a single process (64.01 seconds). Python 3.13.5/Linux, 801 measured package statements, 46 uncovered, **94% rounded coverage**. Controlled thread limits, Matplotlib Agg and disabled pytest plugin autoload were used. Eight newly added tests exercise the audit findings; the old version-specific test assertion was updated to v1.8.0.

**Previously observed for v1.7.0 (historical):** 182 passing offline tests, 0 failures, 1 network-only test deselected across three complete coverage-instrumented groups (66 + 60 + 56). Combined package coverage: 791 statements, 46 uncovered, **94%**. Python 3.13.5/Linux with BLAS thread caps, Matplotlib Agg and pytest plugin autoload disabled. Fifteen new tests specifically cover this audit. The earlier monolithic pytest attempt timed out; the completed three-group runs are the source of this count. GitHub remains untouched pending explicit publication approval.

The previous v1.8.0 verification report describes its own earlier archive, not the current release.

The following **external gates are still outstanding**:

- Real OpenML MNIST historical and official experiments (the original 96% is historical report-only evidence).
- Ruff lint ruleset execution on actual release (if Ruff is unavailable locally, CI remains configured but unverified).
- Hosted GitHub Actions across Python 3.10, 3.12 and 3.13 (requires publication approval).
- Actual Python 3.13/Linux wheelhouse acquisition, creation of a true hash-locked file, and offline hash-locked installation.

SHA-256 checks verify content against the reviewed local release allowlist but are not signatures of upstream authenticity. Synthetic results are never labeled as observed MNIST results.
