"""Data identity hashes and environment fingerprints; no source attestation.

A digest identifies supplied bytes but does not authenticate an upstream source.
The application never issues in-process OpenML identity credentials.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

OPENML_LOADER_SOURCE = "openml_loader_observed_unverified"

# These are labels of *unverified input classes*, never upstream certificates.
# Free-form caller descriptions remain separate, explicitly untrusted notes.
UNVERIFIED_INPUT_LABELS = frozenset(
    {
        "user_supplied_unverified_prepared_split",
        "user_supplied_unverified_array",
        "user_supplied_unverified_npz",
        "unverified_array_input",
        "synthetic_test_fixture",
        "synthetic_fixture",
        "synthetic",
        "fixture",
    }
)


def source_label(source: str, *, prepared: bool = False) -> str:
    """Return only a reviewed, unverified provenance category for saved reports."""
    validate_caller_source(source)
    if prepared:
        return "user_supplied_unverified_prepared_split"
    # File transport cannot be inferred from a caller-authored text label.
    if source in {"user_supplied_unverified_npz", "user_supplied_unverified_prepared_split"}:
        return "user_supplied_unverified_array"
    if source in UNVERIFIED_INPUT_LABELS:
        return source
    return "user_supplied_unverified_array"


def validate_caller_source(source: str) -> str:
    """Treat every user-origin source label as unverified explanatory metadata.

    Reserved acquisition labels cannot be supplied to array APIs. This is not
    cryptographic authentication: arbitrary caller text is also not trusted.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty string")
    if source.strip().lower().startswith("openml_") or source == OPENML_LOADER_SOURCE:
        raise ValueError("reserved OpenML provenance loader source cannot be set through a caller-array API")
    return source


def hash_arrays(*arrays: np.ndarray) -> str:
    """Include dtype, shape, and C-contiguous bytes to prevent aliasing."""
    digest = hashlib.sha256()
    for arr in arrays:
        value = np.asarray(arr)
        if value.dtype.kind not in "iufbUS":
            raise ValueError("Object, structured and pointer-based arrays cannot be hashed reproducibly")
        # Long-double buffers may contain arbitrary padding bytes (e.g. x87
        # 80-bit floats held in 16-byte slots). Hash a canonical float64 wire
        # value when exactly representable; reject lossy values explicitly.
        if value.dtype.kind == "f" and value.dtype.itemsize > 8:
            converted = value.astype(np.float64)
            if not np.array_equal(converted.astype(value.dtype), value):
                raise ValueError("extended floats are not exactly representable for portable hashing")
            value = converted
        # Normalize byte order for stable cross-platform byte fingerprints.
        if value.dtype.byteorder not in ("|", "<"):
            value = value.astype(value.dtype.newbyteorder("<"), copy=False)
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _source_checkout_root(package: Path) -> Path | None:
    """Accept only the expected distribution's source tree as a checkout.

    A surrounding Git repository and an unrelated pyproject.toml cannot
    establish ownership of the MNIST package. Installed-only copies are
    fingerprinted as a package even if a foreign project vendors them.
    """
    source_dir = package.parent
    if source_dir.name != "src" or package.name != "mnist_optimization":
        return None
    candidate = source_dir.parent
    metadata_path = candidate / "pyproject.toml"
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("rb") as handle:
            project_metadata = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(project_metadata, dict):
        return None
    project = project_metadata.get("project")
    if not isinstance(project, dict) or project.get("name") != "mnist-logistic-optimization":
        return None
    return candidate


def code_fingerprint() -> str:
    """Hash stable relative names and bytes, independent of install location.

    A source checkout covers package code, scripts and project metadata. An
    installed wheel only contains its package modules. The environment report
    names this scope; fingerprints from differing scopes must not be compared.
    """
    package = Path(__file__).resolve().parent
    project = _source_checkout_root(package)
    members = [(f"mnist_optimization/{path.name}", path) for path in package.glob("*.py")]
    if project is not None:
        members.extend((f"scripts/{path.name}", path) for path in (project / "scripts").glob("*.py"))
        members.append(("pyproject.toml", project / "pyproject.toml"))
    digest = hashlib.sha256()
    for relative, path in sorted(members):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_fingerprint() -> dict:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    package = Path(__file__).resolve().parent
    root = _source_checkout_root(package)
    revision = None
    git_state = "not_git_checkout"
    tracked_sources = False
    if root is not None:
        try:
            top_level = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout.strip()
            if Path(top_level).resolve() == root.resolve():
                # The commit must actually contain *all* executed modules and
                # scripts; a matching project name alone proves nothing.
                required = sorted(
                    [
                        *(f"src/mnist_optimization/{file.name}" for file in package.glob("*.py")),
                        *(f"scripts/{file.name}" for file in (root / "scripts").glob("*.py")),
                        "pyproject.toml",
                    ]
                )
                tracked = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", "--", *required],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                tracked_sources = tracked.returncode == 0
                if not tracked_sources:
                    git_state = "untracked_source"
                else:
                    status = subprocess.run(
                        ["git", "status", "--porcelain", "--untracked-files=all", "--", *required],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=True,
                    ).stdout
                    if status.strip():
                        git_state = "dirty_source"
                    else:
                        candidate = subprocess.run(
                            ["git", "rev-parse", "HEAD"],
                            cwd=root,
                            capture_output=True,
                            text=True,
                            timeout=3,
                            check=True,
                        ).stdout.strip()
                        if len(candidate) in (40, 64) and all(c in "0123456789abcdef" for c in candidate):
                            revision = candidate
                            git_state = "clean"
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            git_state = "git_verification_unavailable"
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "matplotlib": version("matplotlib"),
        "scikit_learn": version("scikit-learn"),
        "git_revision": revision,
        "git_revision_state": git_state,
        "git_executed_sources_tracked": tracked_sources,
        "source_sha256": code_fingerprint(),
        "source_fingerprint_scope": ("package_scripts_pyproject" if root is not None else "installed_package_only"),
        "blas_lapack_config": str(getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}))[:1000],
    }
