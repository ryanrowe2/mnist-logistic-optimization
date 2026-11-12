# First-order optimization: geometry, code audit, and experiments

## Historical findings from MATH 173A HW 4

The 2023 supplemental assignment specifies a comparison of Euclidean, L-infinity, and L1-steepest descent directions on 4-versus-9 MNIST. It displays a coordinate-update frequency heatmap. Reviewing the original code against the equations exposed two discrepancies:

**1. The L-infinity direction.** The assignment specifies \(d=\|g\|_1\operatorname{sign}(g)\). The historical code uses `np.sign(grad) / grad_l1_norm`: **division instead of multiplication**. With \(\eta=10^{-8}\), the historical loss of roughly `0.69304718` after 10,000 updates is close to the zero-weight baseline \(\log 2 \approx 0.69314718\). Those observations describe the historical *implemented* algorithm; they do **not** validate the theoretically specified direction.

**2. The L1/coordinate direction.** The assignment specifies \(d=\operatorname{sign}(g_j)\|g\|_\infty e_j = g_j e_j\), for a maximum-magnitude coordinate \(j\). The submitted code instead used `np.sign(grad[j_star]) * e_j`, omitting the gradient magnitude. This is a constant-size signed coordinate step, not the stated norm-steepest direction.

**3. No controlled comparison.** The historical comparisons use differing update scales and step sizes and do not control stopping criteria or wall-clock effort. The reported old end losses (approximately L2 `0.0002795`, L1 `0.0013347`, L-infinity `0.6930472`) cannot justify a general ranking of optimizer quality.

**4. Feature importance.** A coordinate-update heatmap records *optimizer selection frequency*, not causal feature importance or a valid independent feature-selection criterion. Gradient magnitude is influenced by pixel scale, correlation, current weights, and the selected direction rule. No ground-truth claim of salience is inferred from update counts.

## Modern reconstructed rules

For \(g=\nabla F(w)\), all start at \(w_0=0\), unless a caller supplies initial weights.

- **Euclidean:** \(d=g\), \(w_{t+1}=w_t-\eta g\).
- **L-infinity geometry:** \(d=\|g\|_1\mathrm{sign}(g)\).
- **L1 geometry / greedy coordinate direction:** \(j^* = \mathrm{argmax}_j |g_j|\) (smallest index on ties); \(d=g_{j^*}e_{j^*}\).

For exactly zero gradients, all directions are zero and no coordinate is selected. `selection_counts` records nonzero-gradient coordinate selections; `update_counts` records only effective, floating-point-representable weight changes. A mathematically nonzero step can round to no change, so the counters can differ. The `coordinate_updates.png` heatmap and `coordinate_update_total` use **effective updates**, while `coordinate_selection_total` records attempted coordinate selections. On a stationary dataset both totals are zero; neither counter is a causal feature-importance estimate.

The historical code labels the L1-direction routine “coordinate descent.” Here this means a *greedy coordinate-direction gradient update*, not exact minimization along a coordinate. Convergence rates, correct learning-rate scales, and runtime costs depend on the geometry and data conditioning.

## Executed offline validation

The checked-in [`figures/synthetic_descent_comparison.png`](../figures/synthetic_descent_comparison.png) and [`results/synthetic_validation.json`](../results/synthetic_validation.json) come from `scripts/generate_synthetic_validation.py` applied to a small, explicitly documented **synthetic two-feature** binary dataset. They test loss reduction and compare the directions under the **same nominal step size**. Neither asset is MNIST or evidence of MNIST classification accuracy. They illustrate that changing the direction geometry changes loss trajectories even when learning rate and initial weights are held fixed.

## Correct experimental comparisons on MNIST

To compare methods using actual MNIST, run each with the same chosen protocol/seed, data matrix, iteration count, and held-out set. Report step-size sensitivity separately, rather than treating a single common nominal \(\eta\) as a fair optimization-tuning procedure across different gradient magnitudes. Do not tune step sizes on evaluation/test data. For rigorous comparisons, select hyperparameters on a *train-only* validation subset, then evaluate the fixed selection once on test. This **train-only hyperparameter selection workflow is now implemented** in `src/mnist_optimization/tuning.py` with `scripts/tune_optimizers.py` and tested on synthetic fixtures. Its execution on genuine MNIST remains pending because OpenML cannot be reached here. The original untuned comparison script remains a clearly labeled illustrative mode.

## Runnable MNIST comparison

The [`compare_optimizers.py`](../scripts/compare_optimizers.py) script runs all three algorithms against one **identical train/evaluation split** and writes individual JSON results, individual numerical traces, and an overlay plot. Its three learning rates must be passed explicitly. Test scores are recorded only for evaluation; they are not used to choose the rates. No MNIST-specific comparison has been executed in the offline build.

## New independent train-only tuning experiment

For each method, configure a candidate grid of positive steps. A seeded, class-stratified split is drawn **only from the selected training rows**. Every candidate receives the same configured number of batch-gradient evaluations. The best validation logistic objective selects the step (smallest step on exact ties); the selected method/step is then refitted from zero on **all** original training rows. Only after all three selections have been made are the held-out evaluation rows accessed for final scores and figure generation.

The archive contains the candidate objectives, selected steps, train/validation source indices, result hashes and a step-sensitivity plot. Distinct geometry-dependent computational costs remain; the fixed gradient-evaluation budget is not a fixed wall-time or floating-point-operation budget. These experiments are new and not a reconstruction of the 2023/2024 notebook.
