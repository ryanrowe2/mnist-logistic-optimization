"""Independent, stratified train-only step-size selection for three geometries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import check_output_available, stage_output, write_json
from .data import prepare_binary, validate_mnist
from .experiments import Configuration, run_dataset, validate_configuration
from .logistic import logistic_loss, validate_real_parameter
from .optimizers import optimize
from .provenance import hash_arrays, source_label, validate_caller_source

METHODS = ("gd", "linf", "coordinate")


def validate_grids(step_grids: dict[str, list[float]]) -> None:
    if set(step_grids) != set(METHODS):
        raise ValueError("step grids must contain gd, linf and coordinate")
    for method in METHODS:
        values = step_grids[method]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{method}: grid must contain finite positive steps")
        for value in values:
            try:
                validate_real_parameter(value, f"{method} step", positive=True)
            except ValueError as exc:
                raise ValueError(f"{method} grid must contain finite positive steps") from exc


def train_validation_indices(y: np.ndarray, *, seed: int, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Stratify the *training array only*, with a seeded independent RNG."""
    fraction = validate_real_parameter(fraction, "validation fraction", positive=True)
    if fraction >= 1:
        raise ValueError("validation fraction must be in (0,1)")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative Python integer, not bool or a mutable RNG")
    labels = np.asarray(y)
    if (labels.ndim != 1 or labels.dtype.kind not in "iuf"
            or not np.all(np.isfinite(labels)) or not np.all(np.isin(labels, (-1, 1)))):
        raise ValueError("training labels must be real, binary -1/+1")
    # A single row-keyed random priority is independent of the sign encoding.
    # Each class picks the rows with the smallest priorities: flipping all labels
    # exchanges class names, never the selected training/validation observations.
    rng = np.random.default_rng(seed)
    priority = np.empty(len(labels), dtype=np.int64)
    priority[rng.permutation(len(labels))] = np.arange(len(labels), dtype=np.int64)
    validation = []
    training = []
    for label in (-1, 1):
        positions = np.flatnonzero(labels == label)
        if len(positions) < 2:
            raise ValueError("train-only validation requires at least two examples per class")
        order = positions[np.argsort(priority[positions])]
        n_val = min(len(positions) - 1, max(1, int(round(len(positions) * fraction))))
        validation.extend(order[:n_val])
        training.extend(order[n_val:])
    return np.asarray(sorted(training), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def tune_and_evaluate(
    X: np.ndarray, y: np.ndarray, *, config: Configuration,
    step_grids: dict[str, list[float]], output: str | Path,
    source: str = "user_supplied_unverified_array",
    validation_fraction: float = 0.2,
) -> dict:
    """Choose each method's step on a training subset; test is untouched until refit."""
    validate_configuration(config)
    validate_grids(step_grids)
    step_grids = {method: [validate_real_parameter(step, f"{method} step", positive=True)
                           for step in steps] for method, steps in step_grids.items()}
    validation_fraction = validate_real_parameter(validation_fraction, "validation fraction", positive=True)
    if validation_fraction >= 1:
        raise ValueError("validation fraction must be in (0,1)")
    validate_caller_source(source)
    safe_source = source_label(source)
    check_output_available(Path(output))
    X, y = validate_mnist(X, y)
    split = prepare_binary(
        X, y, pair=config.digits, protocol=config.protocol,
        train_per_class=config.train_per_class, test_per_class=config.test_per_class, seed=config.seed,
    )
    tr, va = train_validation_indices(split.y_train, seed=config.seed + 1, fraction=validation_fraction)
    with stage_output(output) as stage:
        candidate_table = {}
        selected = {}
        for method in METHODS:
            attempts = []
            for step in step_grids[method]:
                try:
                    result = optimize(
                        split.X_train[tr], split.y_train[tr], method=method,
                        step_size=step, iterations=config.iterations,
                        record_every=max(1, config.record_every),
                        l2=config.l2, bias_index=split.bias_index,
                    )
                    score = logistic_loss(
                        result.weights, split.X_train[va], split.y_train[va],
                        l2=config.l2, bias_index=split.bias_index,
                    )
                except FloatingPointError as exc:
                    attempts.append({"step_size": step, "validation_objective": None, "error": str(exc)})
                else:
                    attempts.append({"step_size": step, "validation_objective": score})
            eligible = [a for a in attempts if a["validation_objective"] is not None]
            if not eligible:
                raise ValueError(f"no numerically valid hyperparameter candidate for {method}")
            winner = min(eligible, key=lambda a: (a["validation_objective"], a["step_size"]))
            selected[method] = winner["step_size"]
            candidate_table[method] = {"candidates": attempts, "selected_step": winner["step_size"]}
        # The entire model-selection phase above has not read split.X_test or split.y_test.
        final = {}
        for method in METHODS:
            fit = replace(config, method=method, step_size=selected[method])
            report = run_dataset(X, y, fit, output=stage / method, source=source)
            final[method] = {
                "selected_step": selected[method],
                "test_accuracy": report["evaluation"]["accuracy"],
                "test_errors": report["evaluation"]["errors"],
                "training_objective": report["optimization"]["final_training_objective"],
            }
        summary = {
            "schema_version": 1,
            "source": safe_source, "user_supplied_source_note_unverified": source,
            "verified_openml": False,
            "acquisition_observation": "not_observed_by_array_api",
            "acquisition_observation_is_authenticated": False,
            "dataset_sha256": hash_arrays(X, y),
            "protocol": config.protocol, "digit_pair": list(config.digits),
            "training_n": len(split.y_train), "validation_n": len(va),
            "selection_training_n": len(tr), "test_n": len(split.y_test),
            "validation_fraction": validation_fraction,
            "seed": config.seed, "selection_seed": config.seed + 1,
            "iterations_per_candidate": config.iterations,
            "matching_budget": "each candidate receives exactly the configured number of full-batch gradients",
            "selection_metric": "minimum train-only held-out regularized logistic objective",
            "selection": candidate_table,
            "final_refit_and_test_once_per_method": final,
            "limitations": (
                "No general superiority inferred; comparisons vary in vector arithmetic "
                "and computational cost."
            ),
        }
        np.savez_compressed(
            stage / "train_validation_rows.npz",
            training_rows=tr, validation_rows=va,
            training_source_indices=split.train_indices[tr],
            validation_source_indices=split.train_indices[va],
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        try:
            for method in METHODS:
                valid = [c for c in candidate_table[method]["candidates"] if c["validation_objective"] is not None]
                ax.plot([c["step_size"] for c in valid],
                        [c["validation_objective"] for c in valid], "o-", label=method)
            ax.set_xscale("log")
            ax.set(xlabel="Step size (log scale)", ylabel="Train-only validation objective",
                   title="Step-size sensitivity — no test-set tuning")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(stage / "validation_steps.png", dpi=160)
        finally:
            plt.close(fig)
        write_json(stage / "selection.json", summary)
    return summary
