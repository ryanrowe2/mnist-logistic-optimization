"""Reconstruct historical 4-versus-9 experiment (OpenML v1, raw pixels, 10k GD)."""

import sys

from mnist_optimization.experiments import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--protocol",
                "historical",
                "--digits",
                "4",
                "9",
                "--train-per-class",
                "500",
                "--test-per-class",
                "500",
                "--optimizer",
                "gd",
                "--step-size",
                "0.0001",
                "--iterations",
                "10000",
                "--output",
                "results/historical_4_vs_9",
            ]
            + sys.argv[1:]
        )
    )
