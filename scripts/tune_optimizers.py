"""Stratified train-only tuning, then one official held-out test per optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mnist_optimization.artifacts import check_output_available
from mnist_optimization.data import load_mnist, validate_mnist
from mnist_optimization.experiments import Configuration, validate_configuration
from mnist_optimization.tuning import tune_and_evaluate, validate_grids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("historical", "official"), default="official")
    parser.add_argument("--digits", nargs=2, type=int, default=(4, 9))
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--gd-grid", nargs="+", type=float, default=[0.05, 0.1, 0.3])
    parser.add_argument("--linf-grid", nargs="+", type=float, default=[0.0001, 0.0005, 0.001])
    parser.add_argument("--coordinate-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--local-npz", type=Path)
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/train_only_tuning"))
    options = parser.parse_args(argv)
    config = Configuration(
        protocol=options.protocol,
        digits=tuple(options.digits),
        train_per_class=options.train_per_class,
        test_per_class=options.test_per_class,
        iterations=options.iterations,
        record_every=options.record_every,
        seed=options.seed,
        l2=options.l2,
    )
    steps = {
        "gd": options.gd_grid,
        "linf": options.linf_grid,
        "coordinate": options.coordinate_grid,
    }
    validate_configuration(config)
    validate_grids(steps)
    if not np.isfinite(options.validation_fraction) or not 0 < options.validation_fraction < 1:
        parser.error("validation-fraction must be in (0,1)")
    check_output_available(options.output)
    if options.local_npz is None:
        X, y = load_mnist(data_home=options.data_home)
        source = "unverified_array_input"
        observation = "openml_loader_invoked_by_cli_unverified"
    else:
        with np.load(options.local_npz, allow_pickle=False) as data:
            X, y = validate_mnist(data["X"], data["y"])
        source = "user_supplied_unverified_npz"
        observation = "local_npz_loaded_by_cli_unverified"
    report = tune_and_evaluate(
        X,
        y,
        config=config,
        step_grids=steps,
        output=options.output,
        source=source,
        validation_fraction=options.validation_fraction,
    )
    # Only this command-line function can report what it actually invoked.
    # Do not pass this observation into a public array-processing function.
    print(
        json.dumps(
            {
                "cli_only_acquisition_observation": observation,
                "final_refit_and_test_once_per_method": report["final_refit_and_test_once_per_method"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
