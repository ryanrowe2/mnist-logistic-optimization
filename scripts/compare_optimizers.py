"""Compare three NumPy direction geometries on one unchanged binary split.

Provided steps are not tuned; see tune_optimizers.py for train-only selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_optimization.artifacts import check_output_available, stage_output, write_json
from mnist_optimization.data import load_mnist, prepare_binary, validate_mnist
from mnist_optimization.experiments import Configuration, run_dataset, validate_configuration
from mnist_optimization.logistic import validate_real_parameter
from mnist_optimization.provenance import hash_arrays, source_label, validate_caller_source


def validate_steps(steps: dict[str, float]) -> None:
    if set(steps) != {"gd", "linf", "coordinate"}:
        raise ValueError("step_sizes must provide gd, linf, and coordinate")
    for value in steps.values():
        validate_real_parameter(value, "method step", positive=True)


def compare(
    X: np.ndarray, y: np.ndarray, *, config: Configuration,
    step_sizes: dict[str, float], output: Path, source: str,
) -> dict:
    """Write nothing unless all methods, figures, and summaries complete."""
    validate_configuration(config)
    validate_steps(step_sizes)
    step_sizes = {method: validate_real_parameter(value, f"{method} step", positive=True)
                  for method, value in step_sizes.items()}
    validate_caller_source(source)
    safe_source = source_label(source)
    check_output_available(output)
    X, y = validate_mnist(X, y)
    prepare_binary(
        X, y, pair=config.digits, protocol=config.protocol,
        train_per_class=config.train_per_class, test_per_class=config.test_per_class,
        seed=config.seed,
    )
    dataset_hash = hash_arrays(X, y)
    with stage_output(output) as stage:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        try:
            table = {}
            for method in ("gd", "linf", "coordinate"):
                current = replace(config, method=method, step_size=step_sizes[method])
                directory = stage / method
                report = run_dataset(X, y, current, output=directory, source=source)
                with np.load(directory / "numeric_results.npz", allow_pickle=False) as values:
                    ax.plot(values["iteration"], values["objective"],
                            label=f"{method} (eta={step_sizes[method]:g})")
                table[method] = {
                    "step_size": step_sizes[method],
                    "final_training_objective": report["optimization"]["final_training_objective"],
                    "evaluation_accuracy": report["evaluation"]["accuracy"],
                    "test_error_count": report["evaluation"]["errors"],
                }
            ax.set(xlabel="Completed gradient updates", ylabel="Training objective",
                   title=f"Directional geometry comparison | {config.protocol} | {safe_source}")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(stage / "comparison.png", dpi=160)
        finally:
            plt.close(fig)
        summary = {
            "source": safe_source, "user_supplied_source_note_unverified": source,
            "dataset_sha256": dataset_hash,
            "acquisition_observation": "not_observed_by_array_api",
            "acquisition_observation_is_authenticated": False,
            "provenance_verified": False, "protocol": config.protocol,
            "digits": list(config.digits), "train_per_class": config.train_per_class,
            "test_per_class": config.test_per_class, "seed": config.seed,
            "iterations": config.iterations, "l2": config.l2,
            "caveat": "Steps are fixed in advance; no test-set tuning, no claim of fair hyperparameters.",
            "methods": table,
        }
        write_json(stage / "comparison.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gd-step", type=float, required=True)
    parser.add_argument("--linf-step", type=float, required=True)
    parser.add_argument("--coordinate-step", type=float, required=True)
    parser.add_argument("--protocol", choices=("historical", "official"), default="historical")
    parser.add_argument("--digits", nargs=2, type=int, default=(4, 9))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/optimizer_comparison"))
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--local-npz", type=Path, default=None)
    options = parser.parse_args(argv)
    steps = {"gd": options.gd_step, "linf": options.linf_step, "coordinate": options.coordinate_step}
    config = Configuration(
        protocol=options.protocol, digits=tuple(options.digits),
        train_per_class=options.train_per_class, test_per_class=options.test_per_class,
        seed=options.seed, iterations=options.iterations, record_every=25, l2=options.l2,
    )
    validate_configuration(config)
    validate_steps(steps)
    check_output_available(options.output)
    if options.local_npz is None:
        X, y = load_mnist(data_home=options.data_home)
        # No array-input API can attest an OpenML acquisition; this CLI invoked
        # the fetch, but downstream numerical reports remain unverified inputs.
        source = "unverified_array_input"
        observation = "openml_loader_invoked_by_cli_unverified"
    else:
        with np.load(options.local_npz, allow_pickle=False) as dataset:
            X, y = validate_mnist(dataset["X"], dataset["y"])
        source = "user_supplied_unverified_npz"
        observation = "local_npz_loaded_by_cli_unverified"
    summary = compare(
        X, y, config=config, step_sizes=steps,
        output=options.output, source=source,
    )
    # This observation is emitted by the CLI after the loader returned, never
    # supplied to the public array-processing API or saved as array provenance.
    print(json.dumps({**summary, "cli_only_acquisition_observation": observation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
