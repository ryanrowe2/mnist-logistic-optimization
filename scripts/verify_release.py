"""Independently verify every archived file hash, including completeness and topology."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT = "mnist-logistic-optimization"
SPEC_FILE = Path(__file__).resolve().parents[1] / "release-files.json"
MAX_MEMBERS = 250
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1000


def expected_release_files() -> set[str]:
    """Read the *local source-controlled* allowlist, not the untrusted ZIP's list."""
    inventory = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    names = inventory.get("files")
    if (inventory.get("schema") != 1 or inventory.get("project") != PROJECT
            or not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names)
            or len(names) != len(set(names))):
        raise ValueError("invalid local release inventory")
    allowed = set(names)
    if "release-files.json" not in allowed:
        raise ValueError("release inventory must include itself")
    return allowed


HASH_LINE = re.compile(r"^([a-f0-9]{64})  (.+)$")


def verify_directory(root: Path) -> None:
    root = Path(root)
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ValueError("symlink is not permitted in the release directory path or its ancestors")
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError("missing MANIFEST.sha256 at project root")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = HASH_LINE.fullmatch(line)
        if match is None:
            raise ValueError("invalid MANIFEST.sha256 syntax")
        digest, name = match.groups()
        pure = PurePosixPath(name)
        if (pure.is_absolute() or name != pure.as_posix() or ".." in pure.parts
                or "\\" in name or len(pure.parts) == 0 or name == "MANIFEST.sha256"):
            raise ValueError("unsafe or self-referencing manifest path")
        if name in expected:
            raise ValueError("duplicate manifest path")
        expected[name] = digest
    actual: dict[str, Path] = {}
    actual_directories: set[str] = set()
    for file in root.rglob("*"):
        if file.is_symlink():
            raise ValueError("symlink is not permitted in a release directory")
        if file.is_file() and file != manifest:
            actual[file.relative_to(root).as_posix()] = file
        elif file.is_dir():
            actual_directories.add(file.relative_to(root).as_posix())
        elif not file.is_file():
            raise ValueError("unexpected special filesystem entry")
    allowed = expected_release_files()
    if set(expected) != allowed:
        raise ValueError("manifest does not match the source-controlled release allowlist")
    if set(expected) != set(actual):
        raise ValueError("missing, unexpected or unlisted release files")
    expected_directories: set[str] = set()
    for name in expected:
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if expected_directories != actual_directories:
        raise ValueError("unexpected, empty or missing release directory")
    for name, path in actual.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise ValueError(f"SHA-256 mismatch: {name}")


def verify_archive(zip_path: Path) -> None:
    """Reject noncanonical, aliased, symlinked and unlisted ZIP entries before extraction."""
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if not names or len(names) != len(set(names)):
            raise ValueError("empty or duplicate archive paths")
        if len(members) > MAX_MEMBERS or sum(member.file_size for member in members) > MAX_TOTAL_BYTES:
            raise ValueError("archive size or member count exceeds safety limits")
        for member in members:
            name = member.filename
            relative = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (member.file_size > MAX_MEMBER_BYTES
                    or member.file_size > MAX_COMPRESSION_RATIO * max(member.compress_size, 1)):
                raise ValueError(f"archive expansion limits exceeded: {name}")
            if (name.endswith("/") or name != relative.as_posix()
                    or relative.is_absolute() or ".." in relative.parts
                    or len(relative.parts) < 2 or relative.parts[0] != PROJECT
                    or "\\" in name or ":" in name or member.create_system != 3
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or member.compress_type != ZIP_DEFLATED or (member.flag_bits & 0x1)):
                raise ValueError(f"unsafe, noncanonical, or nonregular archive entry: {name}")
        prefix = f"{PROJECT}/"
        manifest_name = prefix + "MANIFEST.sha256"
        if manifest_name not in names or prefix + "pyproject.toml" not in names:
            raise ValueError("missing repository-root metadata")
        inventory: set[str] = set()
        content = archive.read(manifest_name).decode("utf-8")
        for line in content.splitlines():
            match = HASH_LINE.fullmatch(line)
            if not match:
                raise ValueError("invalid archive manifest syntax")
            _digest, name = match.groups()
            normalized = PurePosixPath(name)
            if (name != normalized.as_posix() or normalized.is_absolute()
                    or ".." in normalized.parts or "\\" in name
                    or name == "MANIFEST.sha256" or name in inventory):
                raise ValueError("unsafe or duplicate archive manifest entry")
            inventory.add(name)
        if inventory != expected_release_files():
            raise ValueError("archive manifest does not match reviewed release allowlist")
        if set(names) != {prefix + name for name in inventory} | {manifest_name}:
            raise ValueError("archive has unlisted, missing, or aliased files")
        with tempfile.TemporaryDirectory(prefix="mnist-release-verify-") as temporary:
            archive.extractall(temporary)
            verify_directory(Path(temporary) / PROJECT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="extracted project directory or release ZIP")
    args = parser.parse_args()
    if args.path.is_file() and args.path.suffix == ".zip":
        verify_archive(args.path)
    else:
        verify_directory(args.path)
    print("Release files, paths and SHA-256 manifest: VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
