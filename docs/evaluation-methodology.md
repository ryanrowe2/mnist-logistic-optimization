# Data, validation methodology, and uncertainty

## Dataset identity

Source: [OpenML MNIST data 554, v1](https://www.openml.org/search?id=554&type=data), scikit-learn `fetch_openml(data_id=554, as_frame=False, return_X_y=True)`. The full MNIST contains 70,000 (28×28) grayscale images. The canonical first 60,000 rows are the official training partition, and the final 10,000 rows are the official test partition. The library is used solely for data loading and local caching, never model optimization.

The supplied arrays are validated for expected dimensions, label ranges and integer pixel values. The standard run does not redistribute dataset bytes. A local `.npz` can be provided but its contents cannot be authenticated merely from shape and range; the source is therefore recorded as **unverified**.

## Historical protocol

For each of the two requested digits, select **its first 500 occurrences** in all 70k OpenML rows for training and its **next 500 occurrences** for evaluation; concatenate class groups and shuffle each independently. The seed is 2024 for determinism in this reconstruction; the original author did not supply a fixed seed. Training and evaluation indices are disjoint, but the evaluation set is **not** MNIST's official test set and cannot be equated with it. Use raw grayscale intensities and no intercept/penalty. Preserve 10,000 GD steps of size 10⁻⁴.

## Improved protocol

Sample the desired number of observations per class from the canonical 60k training partition, without replacement. The six independent RNGs are keyed by actual digit identity rather than assigned by negative/positive class position: selection uses `np.random.SeedSequence([seed, 0, digit])` for training and `np.random.SeedSequence([seed, 1, digit])` for evaluation; shuffling uses `np.random.SeedSequence([seed, 2, low_digit, high_digit])` for training and `[seed, 3, low_digit, high_digit]` for evaluation, where `low_digit, high_digit = sorted(pair)`. The concatenation order is sorted by digit identity before each seeded shuffle. Reversing the requested pair therefore preserves both sets of image rows and their order while negating the labels. Evaluation observations are sampled separately from the official 10k test partition. The train and test streams are independent: changing test sample counts does not change training membership or ordering. Both sets are class-balanced by construction. Divide all intensities by the fixed factor 255 (no learned transform), and append a bias column of ones. Fit the linear weights using only training data. Optional L2 regularization excludes the final bias coefficient.

Default config: 2,000 samples per class for training, 500 per class for final evaluation; seed 2024; 2,000 GD updates with step 0.3 and λ=0.01. These are **configuration defaults**, not scientifically validated/tuned optimum values. For train-only hyperparameter selection, `scripts/tune_optimizers.py` further partitions the training rows with one seeded, row-keyed priority permutation and a stratified selection that does not change under a reversal of binary label signs, then chooses method-specific step sizes using validation logistic loss alone, then refits on all training observations and evaluates test samples only once per method. This is a new experiment, not the 2024 baseline.

## Metrics

`metrics.json` contains two-sided 95% Wilson score intervals for **observed** accuracy and recall (`accuracy_wilson_95_interval`, `recall_wilson_95_by_class`), along with accuracy, error rate, integer counts, true-vs-predicted confusion counts (rows/columns in negative/positive order), class-specific error counts and rates, and class-specific recall. It includes training and test results separately, the full configuration, data indices (first 10 for display; all saved in `numeric_results.npz`), initial/final loss, environment versions, and an enumerated unverified provenance label. Both training and evaluation true labels/predictions are saved in `numeric_results.npz` so both confusion matrices can be recomputed without the source images. The reported objective includes the configured regularizer; the unregularized training loss is recorded separately.

When `n=1000` evaluation examples and a measured accuracy is near 96%, binomial sampling variability alone gives an approximate standard error \(\sqrt{0.96\times0.04/1000}\approx0.0062\), or 0.62 percentage points, assuming independent representative samples. A simple normal 95% interval would be approximately 94.8% to 97.2%; actual interval construction and dependence/selection may differ. **This is uncertainty arithmetic, not a newly measured confidence interval for a rerun.** Further, the historical 4% error was printed with only two decimals.

## Interpretation caveats

Balanced per-class subsets measure conditional performance for the two selected classes; they do not establish ten-digit MNIST performance or deployed class distributions. The original dataset order and class-specific consecutiveness may produce selection effects. Zero training error after 10,000 updates cannot establish robust generalization. For the **historical** protocol, prediction uses the original `np.sign` with a zero tie counted as an unclassified error and a third confusion column when needed. For **official** evaluation, an exact-zero score maps to +1.


## Reproducible result provenance

Run reports include a SHA-256 digest of all 70,000 input source rows for `run_dataset`, the exact prepared sample features and labels, complete selected source indices in the NPZ file, and a source-module digest. No in-process loader token is an authentication credential. The output always marks `canonical_mnist_assertion` and `independent_dataset_identity_verified` false unless a future independently authenticated dataset reference is introduced. The CLI may record a successful `fetch_openml(data_id=554)` *route* and the bytes it received; this is not a cryptographic attestation of the upstream dataset. A root `_SUCCESS.json` file, containing a SHA-256 inventory of every output file plus a complete directory inventory, is written before publication of each immutable result directory. The `metrics.json` report has no volatile timestamp, allowing byte-for-byte comparison in a fixed environment. Note that OpenML transport provenance is not a separate cryptographic signature from the dataset owner.

## Confidence intervals and prepared splits

For `k` correct predictions out of `n` held-out samples and two-sided confidence `1-α=0.95`, Wilson's score interval uses
`z=Φ⁻¹(0.975)`, center `(k/n + z²/(2n))/(1+z²/n)`, and half-width
`z * sqrt((k/n)(1-k/n)/n + z²/(4n²))/(1+z²/n)`.
The same construction applies to each class's recall. These intervals describe binomial sampling uncertainty under simplifying representative/i.i.d. assumptions; the class-balanced, historically ordered design may violate them. No exact interval is retroactively assigned to a rounded historical score.

`run_prepared` can analyze custom arrays, but its metrics intentionally mark the sampling protocol **user-supplied/unverified**. Claims about official or historical source selection are only validated by the full-dataset `prepare_binary` workflow. Official-format prepared inputs must nevertheless satisfy partition index bounds, pixel-range rules, and a unit intercept.

## Untrusted source annotations and completion-file parsing

Arbitrary descriptions passed as `source` by Python callers remain explicitly
marked `user_supplied_source_note_unverified`. Machine-readable `source` fields
and experiment plot titles use an enumerated, *unverified* category; a note
claiming OpenML authenticity cannot change these labels. CLI-only loader
observations are separate and never attest that downloaded bytes are canonical.

The `_SUCCESS.json` manifest is a content integrity inventory, not an upstream
authentication signature. Its validator rejects duplicate JSON object keys
(including nested maps), unexpected top-level schema fields, symlinks, extra
files and directories, and mismatched file digests. The training and test
metrics can be recomputed from the stored `y_train`, `y_pred_train`, `y_test`,
and `y_pred_test` arrays, without reacquiring MNIST.
