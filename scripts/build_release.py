"""Build one portable, reproducible, hash-inventoried repository ZIP.

Safety: no symlink traversal, exclusive temporary files, atomic publication.
Run from any directory: python scripts/build_release.py --output /path/to/release.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "mnist-logistic-optimization"


def release_files() -> tuple[str, ...]:
    """Use the reviewed source-controlled list, never glob arbitrary files."""
    inventory = json.loads((ROOT / "release-files.json").read_text(encoding="utf-8"))
    if inventory.get("schema") != 1 or inventory.get("project") != PROJECT:
        raise ValueError("invalid release inventory header")
    names = inventory.get("files")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("invalid or duplicated release inventory entries")
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or name == "MANIFEST.sha256"
        ):
            raise ValueError("unsafe release inventory entry")
    if not {"release-files.json", "pyproject.toml", "README.md", "scripts/verify_release.py"} <= set(names):
        raise ValueError("release inventory misses required metadata")
    return tuple(names)


RELEASE_FILES = release_files()


def _guard_path(path: Path) -> None:
    """Refuse a symlink at any existing ancestor (including dangling links)."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"symlinked release path is forbidden: {path}")


def included_files(root: Path) -> list[Path]:
    """Only explicitly reviewed regular files are eligible for publication."""
    root = Path(root)
    _guard_path(root)
    paths = []
    for name in RELEASE_FILES:
        file = root / name
        _guard_path(file)
        if not file.is_file():
            raise ValueError(f"missing or non-regular release file: {name}")
        paths.append(file)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def _read_regular(path: Path) -> bytes:
    _guard_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"release entry is not a regular file: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def manifest_contents(root: Path, files: list[Path]) -> bytes:
    """Return a hash inventory using the same no-follow file-reader as the builder."""
    rows = [f"{hashlib.sha256(_read_regular(file)).hexdigest()}  {file.relative_to(root).as_posix()}" for file in files]
    return ("\n".join(rows) + "\n").encode("utf-8")


def build_archive(output: Path, root: Path = ROOT) -> Path:
    # abspath, unlike Path.resolve(), never follows an existing output symlink.
    output = Path(os.path.abspath(output))
    root = Path(os.path.abspath(root))
    _guard_path(root)
    _guard_path(output)
    if output.is_relative_to(root):
        raise ValueError("archive output must be outside the repository root")
    files = included_files(root)
    # Capture each file exactly once: archive bytes and manifest cannot diverge
    # if a source file changes between hashing and ZIP construction.
    snapshots = {file.relative_to(root).as_posix(): _read_regular(file) for file in files}
    manifest = (
        "\n".join(f"{hashlib.sha256(content).hexdigest()}  {name}" for name, content in sorted(snapshots.items()))
        + "\n"
    ).encode("utf-8")
    destination_manifest = root / "MANIFEST.sha256"
    _guard_path(destination_manifest)
    if destination_manifest.exists() and not destination_manifest.is_file():
        raise ValueError("manifest must be a regular file")

    # Exclusive temporary files prevent preexisting symlink redirects. Atomic
    # replacement never opens the destination path through an existing link.
    manifest_fd, manifest_tmp = tempfile.mkstemp(prefix=".manifest-build-", dir=root)
    try:
        with os.fdopen(manifest_fd, "wb") as stream:
            stream.write(manifest)
        _guard_path(destination_manifest)
        os.replace(manifest_tmp, destination_manifest)
    finally:
        Path(manifest_tmp).unlink(missing_ok=True)

    _guard_path(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    _guard_path(output)
    zip_fd, zip_tmp = tempfile.mkstemp(prefix=f".{output.name}-build-", suffix=".zip", dir=output.parent)
    try:
        with os.fdopen(zip_fd, "wb") as stream:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
                for name, content in sorted({**snapshots, "MANIFEST.sha256": manifest}.items()):
                    info = ZipInfo(f"{PROJECT}/{name}", date_time=(2026, 11, 11, 12, 0, 0))
                    info.create_system = 3
                    info.compress_type = ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, content, compress_type=ZIP_DEFLATED, compresslevel=9)
        _guard_path(output)
        os.replace(zip_tmp, output)
    finally:
        Path(zip_tmp).unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_archive(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
