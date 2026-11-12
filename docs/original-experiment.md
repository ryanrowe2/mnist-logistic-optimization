# Original experiment: forensic reconstruction

## Evidence and recovery status

The July 18, 2024, 16-page report is preserved as [`sources/2024-original-report.pdf`](sources/2024-original-report.pdf). The same underlying exercise appears in the 21-page `MATH 173A HW 3.pdf` and the 19-page `MATH 173A HW 4.pdf`, examined separately in the user's source library. The files contain notebook-like execution-cell transcripts, source snippets, rendered plots, and printed results. **No separate original `.ipynb` or `.py` file was recovered** in the available library search. Every module in `src/` is therefore a *modern reconstruction* from the historical record, not a recovered original notebook.

### Directly evidenced original procedure

1. `fetch_openml('mnist_784', version=1, return_X_y=True)` loaded 70,000 rows with 784 pixel features. `y` was converted to NumPy and into per-digit arrays in original relative ordering.
2. For digits A and B, the first 500 per-digit examples became training. Per class, the next 500 (`[500:1000]`) became evaluation. Labels A = -1, B = +1. Each assembled set was randomly permuted using the global, **unseeded** `np.random.shuffle`.
3. Feature intensities were not divided by 255; images were flattened into 784-dimensional rows without an intercept.
4. Weights started at zero; gradient descent used \(\eta=0.0001\) and 10,000 updates, with the mean logistic loss computed after each update.
5. Classification used `np.sign(X @ weights)` and measured the mean mismatch with labels. `f'{error_rate:.2f}'` printed only **two decimal places**.

### Historical output vs verification status

| Run | Original report | Additional source | Status |
|---|---|---|---|
| 0 vs 1 evaluation error | `0.00` (report p. 12) | HW 3 shows `0.01` in a separate run | Historical printed outputs; difference may involve different runs/seeds; not independently rerun |
| 4 vs 9 evaluation error | `0.04` (report p. 16) | HW 3 also shows `0.04` | Historical printed output; implies about 96% accuracy to reported precision |
| 4 vs 9 training error | `0.00` (report p. 16) | HW 3 shows `0.00` | Rounded printed output, **not proof of exactly zero errors** |
| 4 vs 9 examples | 500 of each class train and 500 of each class evaluate | Same | Supported source methodology |
| GD iterations | 10,000 | Same | Supported source methodology |

Because `0.04` is rounded to two decimal places, it **does not prove exactly 40 errors**; exact confusion matrices are unavailable in the report. It does support the résumé's phrase “reached 96% accuracy” at the reported precision.

### Reconstruction choices and evidence gaps

- The exact 2024 permutation is not recoverable because no random seed was recorded. Full-batch mean gradients are insensitive to row order mathematically, but floating-point summation order may affect low-order results. The new code fixes the reconstruction seed at 2024 and makes it configurable.
- OpenML dataset 554 version 1 is specified to retain dataset identity/order, but an offline environment prevented independent MNIST downloads and original-result reruns during preparation.
- Historical experiment mode preserves raw pixel units, 784-dimensional weights, disjoint consecutive per-class slices, fixed GD settings, and per-update loss recording (10,001 checkpoints including the initial loss). It deliberately does not use the official 60k/10k split.
- The original unstable exponential and softplus expressions are replaced with mathematically equivalent stable expressions. Historical mode now deliberately preserves the original `np.sign` zero-tie behavior; the distinct improved official protocol consistently maps ties to +1. This difference is recorded and tested.
- Historical progress plots and printed percentages are preserved inside the PDF, not recreated as if independently measured.
