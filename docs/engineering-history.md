# Engineering provenance and prior audit history

This public release keeps one short audit timeline rather than shipping all previous working notes. The original 2024 academic PDF is preserved byte-for-byte at `sources/2024-original-report.pdf`. The PDF is the historical primary source; this repository’s Python implementation is a modern reconstruction.

| Phase | Substantive correction |
|---|---|
| Initial reconstruction | NumPy logistic objective, gradient, direct optimizers, historical and official partitions, synthetic validation |
| v1.2–1.3 | Transactional experiment directories; input contracts; finite-difference tests; strict artifact inventories and provenance disclaimers |
| v1.4–1.5 | Portable ZIP manifests, explicit file allowlist, numerical edge cases, dataset fingerprint portability, scalar validation |
| v1.6 | ZIP resource bounds, controlled test execution, release allowlist completeness, path safety |
| v1.7 | Independent official training RNG streams; accepted NumPy scalar serialization; result ancestor symlink protection; consistent CLI acquisition observations; documentation consolidation |
| v1.8 | Reject symlinks before output path resolution; restrict API acquisition observations to fixed array-API state; CLI-only route observation; document six RNG streams |
| v1.12 | Verify project metadata before Git attribution; use digit-identity-keyed official sampling; reject non-1-D source indices |

Detailed audit records for versions through 1.8.0 are preserved in the previous publication packages, not shipped as current recruiter-facing claims. Historical accuracy figures are quotations from the academic report, not newly reproduced MNIST results. For current verification evidence see [release-status.md](release-status.md); for limitations see [limitations.md](limitations.md).

