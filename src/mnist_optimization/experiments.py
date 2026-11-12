"""Reproducible experiment runner with validated provenance and atomic artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import check_output_available, stage_output, write_json
from .data import BinarySplit, load_mnist, prepare_binary, validate_mnist
from .evaluation import metrics
from .figures import save_figures
from .logistic import logistic_loss, predict, validate_data, validate_real_parameter
from .optimizers import Method, optimize
from .provenance import (
    OPENML_LOADER_SOURCE, environment_fingerprint, hash_arrays, source_label, validate_caller_source,
)


@dataclass(frozen=True)
class Configuration:
    protocol: str = "historical"
    digits: tuple[int, int] = (4, 9)
    train_per_class: int = 500
    test_per_class: int = 500
    seed: int = 2024
    method: Method = "gd"
    step_size: float = 1e-4
    iterations: int = 10000
    record_every: int = 1
    l2: float = 0.0

    def __post_init__(self) -> None:
        """Store valid NumPy real scalars as JSON-serializable Python floats.

        Invalid values stay untouched until explicit configuration validation,
        preserving the existing caller validation contract.
        """
        if isinstance(self.digits, (tuple, list)) and len(self.digits) == 2:
            if all(type(d) is int and 0 <= d <= 9 for d in self.digits):
                object.__setattr__(self, "digits", tuple(self.digits))
        for field in ("step_size", "l2"):
            value = getattr(self, field)
            if isinstance(value, (np.integer, np.floating)):
                try:
                    number = validate_real_parameter(value, field, positive=field == "step_size")
                except ValueError:
                    continue
                object.__setattr__(self, field, number)


def validate_configuration(config: Configuration) -> None:
    """Fast validation before any OpenML download, file open or optimization."""
    if config.protocol not in ("historical", "official"):
        raise ValueError("protocol must be historical or official")
    if (not isinstance(config.digits, (tuple, list)) or len(config.digits) != 2
            or any(type(d) is not int or d not in range(10) for d in config.digits)
            or config.digits[0] == config.digits[1]):
        raise ValueError("digits must be distinct decimal integers 0..9")
    for field in ("train_per_class", "test_per_class", "iterations", "record_every"):
        v = getattr(config, field)
        if type(v) is not int or v <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if config.method not in ("gd", "linf", "coordinate"):
        raise ValueError("method must be gd, linf, or coordinate")
    validate_real_parameter(config.step_size, "step_size", positive=True)
    validate_real_parameter(config.l2, "l2")


def validate_split(split: BinarySplit, config: Configuration) -> None:
    if split.protocol != config.protocol or split.digit_pair != config.digits or split.seed != config.seed:
        raise ValueError("split and configuration refer to different experiments")
    if len(split.y_train) != 2 * config.train_per_class or len(split.y_test) != 2 * config.test_per_class:
        raise ValueError("split sample counts do not match the recorded configuration")
    for indices, labels in ((split.train_indices, split.y_train), (split.test_indices, split.y_test)):
        if not isinstance(indices, np.ndarray) or indices.ndim != 1:
            raise ValueError("split source indices must be one-dimensional NumPy arrays")
        if (len(indices) != len(labels) or np.unique(indices).size != len(indices)
                or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= 70000)):
            raise ValueError("split indices must be nonnegative and individually unique")
        if any(int(np.sum(labels == label)) != len(labels) // 2 for label in (-1, 1)):
            raise ValueError("split class balance does not match requested per-class counts")
    if np.intersect1d(split.train_indices, split.test_indices).size:
        raise ValueError("split indices must be individually unique and disjoint")
    train, _ = validate_data(split.X_train, split.y_train)
    test, _ = validate_data(split.X_test, split.y_test)
    if len(train) != len(split.y_train) or len(test) != len(split.y_test) or train.shape[1] != test.shape[1]:
        raise ValueError("split features and labels must have compatible shapes")
    if config.protocol == "historical":
        if split.scaling != "raw_0_255" or split.bias_index is not None:
            raise ValueError("historical split must use raw features without an intercept")
    else:
        if (split.scaling != "fixed_divide_255" or split.bias_index != 784
                or train.shape[1] != 785 or test.shape[1] != 785):
            raise ValueError("official split must contain 784 normalized pixels plus an intercept")
        if np.any(split.train_indices >= 60000) or np.any(split.test_indices < 60000):
            raise ValueError("official indices must follow MNIST 60000/10000 train/test boundaries")
        if (not np.all(train[:, -1] == 1) or not np.all(test[:, -1] == 1)
                or np.any(train[:, :784] < 0) or np.any(train[:, :784] > 1)
                or np.any(test[:, :784] < 0) or np.any(test[:, :784] > 1)):
            raise ValueError("official pixel values must lie in [0,1] and intercepts must all equal 1")


def _execute(
    split: BinarySplit, config: Configuration, *, output: str | Path,
    source: str, full_dataset: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    validate_configuration(config)
    # This internal numerical writer cannot create a loader observation from
    # caller-provided metadata. In particular, reserved OpenML identifiers and
    # supplied digest/verification flags are not accepted.
    validate_caller_source(source)
    safe_source = source_label(source, prepared=full_dataset is None)
    validate_split(split, config)
    selection_validated = False
    dataset_sha256 = None
    if full_dataset is not None:
        if not isinstance(full_dataset, tuple) or len(full_dataset) != 2:
            raise ValueError("full_dataset must be a pair of complete X and y arrays")
        full_X, full_y = validate_mnist(*full_dataset)
        # Independently reconstruct the selection from the actual supplied
        # values. No arbitrary BinarySplit may borrow their hash or claim the
        # selection procedure was run.
        expected = prepare_binary(
            full_X, full_y, pair=config.digits, protocol=config.protocol,
            train_per_class=config.train_per_class, test_per_class=config.test_per_class,
            seed=config.seed,
        )
        for field in ("X_train", "y_train", "X_test", "y_test", "train_indices", "test_indices"):
            if not np.array_equal(getattr(split, field), getattr(expected, field)):
                raise ValueError(f"prepared split does not match the supplied full dataset: {field}")
        if (split.digit_pair, split.protocol, split.scaling, split.seed, split.bias_index) != (
            expected.digit_pair, expected.protocol, expected.scaling, expected.seed, expected.bias_index,
        ):
            raise ValueError("prepared split metadata does not match the supplied full dataset")
        dataset_sha256 = hash_arrays(full_X, full_y)
        selection_validated = True
    destination = Path(output)
    check_output_available(destination)
    with stage_output(destination) as stage:
        result = optimize(
            split.X_train, split.y_train, method=config.method, step_size=config.step_size,
            iterations=config.iterations, record_every=config.record_every,
            bias_index=split.bias_index, l2=config.l2,
        )
        historical = config.protocol == "historical"
        train_pred = predict(split.X_train, result.weights, historical_sign=historical)
        test_pred = predict(split.X_test, result.weights, historical_sign=historical)
        report = {
            "report_schema": 2,
            "source": safe_source,
            "user_supplied_source_note_unverified": source,
            "config": asdict(config),
            "data": {
                "protocol": split.protocol if selection_validated else "user_supplied_prepared_unverified",
                "requested_protocol": split.protocol,
                "selection_protocol_validated": selection_validated,
                "train_count": int(len(split.y_train)),
                "test_count": int(len(split.y_test)),
                "feature_count": int(split.X_train.shape[1]),
                "train_class_counts_minus_plus": [int(np.sum(split.y_train == c)) for c in (-1, 1)],
                "test_class_counts_minus_plus": [int(np.sum(split.y_test == c)) for c in (-1, 1)],
                "scaling": split.scaling,
                "bias_index": split.bias_index,
                "train_indices_first_10": split.train_indices[:10].tolist(),
                "test_indices_first_10": split.test_indices[:10].tolist(),
                "train_test_overlap": int(np.intersect1d(split.train_indices, split.test_indices).size),
                "dataset_sha256": dataset_sha256,
                "prepared_split_sha256": hash_arrays(
                    split.X_train, split.y_train, split.X_test, split.y_test,
                    split.train_indices, split.test_indices,
                ),
                "canonical_mnist_assertion": False,
                "independent_dataset_identity_verified": False,
                "provenance_note": (
                    "Selection reconstructed from supplied full-size arrays, with a computed "
                    "dataset digest; acquisition route and canonical MNIST identity unverified."
                    if selection_validated else
                    "Caller-provided prepared split; acquisition route, selection provenance, "
                    "and canonical MNIST identity unverified."
                ),
            },
            "optimization": {
                "initial_loss": float(result.losses[0]),
                "final_training_objective": float(result.losses[-1]),
                "final_training_unregularized_loss": logistic_loss(result.weights, split.X_train, split.y_train),
                "initial_update_count": 0,
                "completed_updates": int(config.iterations),
                "coordinate_update_total": int(result.update_counts.sum()),
                "coordinate_selection_total": int(result.selection_counts.sum()),
            },
            "training": metrics(split.y_train, train_pred),
            "evaluation": metrics(split.y_test, test_pred),
            "environment": environment_fingerprint(),
        }
        write_json(stage / "metrics.json", report)
        np.savez_compressed(
            stage / "numeric_results.npz",
            weights=result.weights,
            iteration=result.iterations,
            objective=result.losses,
            update_counts=result.update_counts,
            selection_counts=result.selection_counts,
            train_indices=split.train_indices,
            test_indices=split.test_indices,
            y_train=split.y_train,
            y_pred_train=train_pred,
            y_test=split.y_test,
            y_pred_test=test_pred,
        )
        save_figures(
            result, split.X_test, split.y_test, test_pred,
            stage / "figures", pair=split.digit_pair, provenance=safe_source,
        )
    return report


def run_prepared(
    split: BinarySplit, config: Configuration, *, output: str | Path,
    source: str = "user_supplied_unverified_prepared_split",
) -> dict:
    """Prepared arrays are always labeled as user supplied, not certified protocol splits."""
    validate_caller_source(source)
    return _execute(split, config, output=output, source=source)


def run_dataset(
    X: np.ndarray, y: np.ndarray, config: Configuration, *, output: str | Path,
    source: str = "user_supplied_unverified_array",
) -> dict:
    validate_configuration(config)
    check_output_available(Path(output))
    # Source text is caller-provided metadata, NEVER a source-identity credential.
    # Do not let an arbitrary array API claim that the loader was executed.
    validate_caller_source(source)
    X, y = validate_mnist(X, y)
    split = prepare_binary(
        X, y, pair=config.digits, protocol=config.protocol,
        train_per_class=config.train_per_class, test_per_class=config.test_per_class,
        seed=config.seed,
    )
    return _execute(
        split, config, output=output, source=source, full_dataset=(X, y),
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("historical", "official"), default="historical")
    parser.add_argument("--digits", nargs=2, type=int, default=(4, 9), metavar=("NEG", "POS"))
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--optimizer", choices=("gd", "linf", "coordinate"), default="gd")
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/latest"))
    parser.add_argument("--data-home", type=Path, default=None, help="scikit-learn OpenML cache")
    parser.add_argument("--local-npz", type=Path, default=None,
                        help="local full-size canonical-order X/y arrays, always marked unverified")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    historical = args.protocol == "historical"
    config = Configuration(
        protocol=args.protocol, digits=tuple(args.digits),
        train_per_class=args.train_per_class if args.train_per_class is not None else (500 if historical else 2000),
        test_per_class=args.test_per_class,
        seed=args.seed, method=args.optimizer,
        step_size=args.step_size if args.step_size is not None else (1e-4 if historical else 0.3),
        iterations=args.iterations if args.iterations is not None else (10000 if historical else 2000),
        record_every=args.record_every if args.record_every is not None else (1 if historical else 50),
        l2=args.l2 if args.l2 is not None else (0.0 if historical else 0.01),
    )
    validate_configuration(config)
    check_output_available(args.output)
    if args.local_npz is not None:
        with np.load(args.local_npz, allow_pickle=False) as dataset:
            X, y = validate_mnist(dataset["X"], dataset["y"])
        source = "user_supplied_unverified_npz"
        report = run_dataset(X, y, config, output=args.output, source=source)
    else:
        X, y = load_mnist(data_home=args.data_home)
        # Observing a local fetch call is CLI-side information, not a property
        # of the user-supplied numerical arrays or an upstream identity proof.
        source = "unverified_array_input"
        observation = OPENML_LOADER_SOURCE
        report = run_dataset(X, y, config, output=args.output, source=source)
    if args.local_npz is not None:
        observation = "local_npz_loaded_by_cli_unverified"
    print(json.dumps({
        "protocol": config.protocol, "source": report["source"],
        "cli_only_acquisition_observation": observation,
        "test_accuracy": report["evaluation"]["accuracy"],
        "test_error": report["evaluation"]["error_rate"],
        "train_accuracy": report["training"]["accuracy"],
        "final_objective": report["optimization"]["final_training_objective"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
