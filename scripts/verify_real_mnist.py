"""Run both complete real-MNIST protocols and commit evidence only on total success.

Requires a network-accessible or cached OpenML dataset 554. No fixture fallback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mnist_optimization.artifacts import check_output_available, stage_output, validate_inventory, write_json
from mnist_optimization.data import load_mnist
from mnist_optimization.experiments import Configuration, run_dataset
from mnist_optimization.provenance import OPENML_LOADER_SOURCE, hash_arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-home", type=Path, default=None)
    args = parser.parse_args(argv)
    # Fail before triggering an expensive network fetch for an invalid destination.
    check_output_available(args.output_root)
    X, y = load_mnist(data_home=args.data_home)
    configurations = {
        "historical_4_9": Configuration(
            protocol="historical",
            digits=(4, 9),
            train_per_class=500,
            test_per_class=500,
            iterations=10000,
            step_size=1e-4,
            record_every=1,
            seed=2024,
            l2=0,
        ),
        "official_4_9": Configuration(
            protocol="official",
            digits=(4, 9),
            train_per_class=2000,
            test_per_class=500,
            iterations=2000,
            step_size=0.3,
            record_every=50,
            seed=2024,
            l2=0.01,
        ),
    }
    with stage_output(args.output_root) as stage:
        summary = {
            "source": OPENML_LOADER_SOURCE,
            "dataset_sha256": hash_arrays(X, y),
            "note": (
                "Observed measurements from the fetch_openml(data_id=554) route; no independently "
                "verified reference digest. Historical shuffle seed is unknown."
            ),
            "runs": {},
        }
        for name, config in configurations.items():
            output = stage / name
            report = run_dataset(X, y, config, output=output, source="unverified_array_input")
            if not validate_inventory(output) or report["data"]["canonical_mnist_assertion"]:
                raise RuntimeError("MNIST experiment artifact verification failed")
            summary["runs"][name] = {
                "accuracy": report["evaluation"]["accuracy"],
                "errors": report["evaluation"]["errors"],
                "accuracy_wilson_95_interval": report["evaluation"]["accuracy_wilson_95_interval"],
                "configuration": report["config"],
            }
        write_json(stage / "full_real_mnist_summary.json", summary)
    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
