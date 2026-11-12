# Scientific and operational limitations

See [maintained release status](release-status.md) for current tests and release gates. This page describes scientific limitations rather than obsolete audit counts.

- The original 2024 report records a rounded 4-versus-9 test error of 0.04 (approximately 96% accuracy on 1,000 evaluation images), but the exact predictions, original notebook and unseeded random state are not recovered. That number is **historical evidence**, not a newly replicated benchmark.
- This package has **not run its live OpenML benchmark in the offline environment**; synthetic fixtures are used for engineering validation and are never evidence of MNIST accuracy.
- Historical class-specific sequential samples differ from the official 60,000/10,000 partition. The modern official experiment uses digit-identity-keyed seeded training and evaluation RNG streams and an intercept with fixed 1/255 scaling.
- The underlying logistic objective is convex; a flat loss curve alone neither proves attainment of a finite global minimizer nor establishes generalization. Separability can prevent an attained unregularized minimizer.
- Fixed iteration counts do not equate computational cost across Euclidean, infinity-norm and coordinate directions. Coordinate selection frequencies do not prove causal feature importance.
- Finite evaluation samples give uncertain accuracy estimates. Report error counts, confusion matrices and confidence intervals, not accuracy alone.
- Model is a binary linear baseline, not a state-of-the-art multiclass MNIST system. Numerical results can vary across NumPy and BLAS versions even with matching source and seeds.
- The single-run CLI records the observational label `openml_loader_observed_unverified`; comparison and tuning array APIs record `not_observed_by_array_api`; the CLI entry points separately print `cli_only_acquisition_observation` to stdout, without verifying identity.
- A downloaded dataset’s SHA-256 identifies supplied bytes but does not authenticate OpenML. No full MNIST dataset is redistributed. Dataset rights remain with its original creators.
- Ruff, hosted CI and platform-specific hashed wheel installs remain unverified until run in an enabled environment.

[Engineering history](engineering-history.md) and the [reproducibility guide](reproducibility.md) describe the provenance and verification workflow.
