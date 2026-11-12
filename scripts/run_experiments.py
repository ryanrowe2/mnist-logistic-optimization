"""Run the improved, official-split 4-versus-9 experiment (distinct from 2024)."""
import sys

from mnist_optimization.experiments import main

if __name__ == "__main__":
    raise SystemExit(main([
        "--protocol", "official", "--digits", "4", "9",
        "--train-per-class", "2000", "--test-per-class", "500",
        "--optimizer", "gd", "--step-size", "0.3", "--l2", "0.01",
        "--iterations", "2000", "--output", "results/official_4_vs_9",
    ] + sys.argv[1:]))
