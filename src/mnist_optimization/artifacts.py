"""Transactional, write-once experiment directories and reproducible inventories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_output_path(destination: str | Path) -> Path:
    """Make an absolute path WITHOUT first following a caller-supplied symlink.

    Reject traversal components: normalizing ``alias/../child`` lexically can
    disagree with the filesystem when ``alias`` is a symbolic link.
    """
    supplied = Path(destination).expanduser()
    if ".." in supplied.parts:
        raise ValueError("output path must not contain parent traversal ('..')")
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if any(item.is_symlink() for item in (absolute, *absolute.parents)):
        raise ValueError("output path and ancestors must not contain symlinks")
    return absolute


def check_output_available(destination: Path) -> None:
    destination = _safe_output_path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Output already contains files: {destination}; choose a new --output directory")


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _tree_contents(folder: Path) -> tuple[dict[str, Path], set[str]]:
    """Inventory a closed tree, rejecting symlinks and non-regular entries.

    Only the root-level `_SUCCESS.json` is omitted; nested files with the same
    basename are ordinary data and must be hashed. Extra empty directories
    are disallowed after completion so an inventory validates the entire tree.
    """
    if any(part.is_symlink() for part in (folder, *folder.parents)) or not folder.is_dir():
        raise ValueError("result root and ancestors must be ordinary non-symlinked directories")
    files: dict[str, Path] = {}
    directories: set[str] = set()
    for entry in folder.rglob("*"):
        rel = entry.relative_to(folder).as_posix()
        if entry.is_symlink():
            raise ValueError(f"symlinks are prohibited in result outputs: {rel}")
        if entry.is_dir():
            directories.add(rel)
        elif entry.is_file():
            if rel != "_SUCCESS.json":
                files[rel] = entry
        else:
            raise ValueError(f"unsupported filesystem entry: {rel}")
    return files, directories


def completion_inventory(folder: Path) -> dict:
    """Commit the root marker last; bind every file and directory in the tree."""
    folder = Path(folder)
    if (folder / "_SUCCESS.json").exists():
        raise FileExistsError("completion marker already exists")
    files, directories = _tree_contents(folder)
    report = {
        "schema_version": 2,
        "status": "complete",
        "file_sha256": {name: sha256_file(path) for name, path in sorted(files.items())},
        "directories": sorted(directories),
    }
    write_json(folder / "_SUCCESS.json", report)
    return report


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    """Refuse ambiguous JSON keys even when the duplicate appears in a nested object."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def validate_inventory(folder: Path) -> bool:
    """Return False for any changed, added, missing, or unsafe filesystem entry."""
    try:
        folder = Path(folder)
        marker = folder / "_SUCCESS.json"
        if marker.is_symlink() or not marker.is_file():
            return False
        report = json.loads(
            marker.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object,
        )
        if (not isinstance(report, dict)
                or set(report) != {"schema_version", "status", "file_sha256", "directories"}
                or type(report.get("schema_version")) is not int or report["schema_version"] != 2
                or report.get("status") != "complete"):
            return False
        hashes = report.get("file_sha256")
        directories = report.get("directories")
        if (not isinstance(hashes, dict) or not isinstance(directories, list)
                or any(not isinstance(name, str) for name in hashes)):
            return False
        actual_files, actual_dirs = _tree_contents(folder)
        if set(actual_files) != set(hashes) or set(directories) != actual_dirs:
            return False
        if any(not isinstance(d, str) for d in directories) or len(directories) != len(actual_dirs):
            return False
        for name, path in actual_files.items():
            digest = hashes[name]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or sha256_file(path) != digest):
                return False
        return True
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError):
        return False


@contextmanager
def stage_output(destination: str | Path) -> Iterator[Path]:
    """Commit only a complete result; refuse to overwrite any nonempty directory.

    An interrupted computation leaves no report in the requested destination.
    The staging directory is created on the same filesystem for atomic rename.
    """
    dest = _safe_output_path(destination)
    check_output_available(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Verify again immediately before allocating the staging directory.
    _safe_output_path(dest)
    stage = Path(tempfile.mkdtemp(prefix=f".{dest.name}-incomplete-", dir=dest.parent))
    try:
        yield stage
        completion_inventory(stage)
        check_output_available(dest)  # Also catches path components swapped to symlinks.
        if dest.exists():
            dest.rmdir()  # An existing empty destination is allowed for test compatibility.
        os.replace(stage, dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
