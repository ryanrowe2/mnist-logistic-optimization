"""Validate a complete local wheelhouse and emit a platform-specific SHA-256 lock.

Install the generated lock in a NEW environment with --no-index, --require-hashes
and --find-links; lock creation alone never certifies an installation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import stat
import sys
import tempfile
import zipfile
from collections import defaultdict
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import parse_tag, sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
MAX_WHEEL_BYTES = 160 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 25000
MAX_RATIO = 2000
READ_CHUNK = 1024 * 1024


def guard_path(path: Path) -> None:
    """Disallow symlinked files and ancestor directories, including dangling links."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"symlink path is forbidden: {path}")


def required_packages(root: Path) -> list[Requirement]:
    """Project runtime, development, and declared validation dependencies."""
    with (root / "pyproject.toml").open("rb") as f:
        document = tomllib.load(f)
    project = document["project"]
    items = list(project["dependencies"])
    # A reproducible build needs the build backend as well as runtime tools.
    items.extend(document.get("build-system", {}).get("requires", []))
    items.extend(project.get("optional-dependencies", {}).get("dev", []))
    validation = root / "requirements-validation.txt"
    if validation.exists():
        items.extend(
            line.strip()
            for line in validation.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    items.extend(("wheel", "packaging", "ruff>=0.8,<1"))
    return [Requirement(item) for item in items]


def _wheel_member_bytes(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> bytes:
    """Read metadata under explicit bounds; verify every member by streaming."""
    if member.file_size > MAX_MEMBER_BYTES:
        raise ValueError("wheel member too large")
    with archive.open(member) as stream:
        data = stream.read(MAX_MEMBER_BYTES + 1)
    if len(data) != member.file_size or len(data) > MAX_MEMBER_BYTES:
        raise ValueError("wheel member expanded beyond allowed bounds")
    return data


def _read_wheel(path: Path, supported_tags: set) -> tuple[str, Version, list[Requirement], str]:
    guard_path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_WHEEL_BYTES:
            raise ValueError(f"wheel size or file type is not permitted: {path.name}")
        name, version, _build, tags = parse_wheel_filename(path.name)
        name = canonicalize_name(name)
        if not (tags & supported_tags):
            raise ValueError(f"wheel is incompatible with this Python/platform: {path.name}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(READ_CHUNK), b""):
                digest.update(chunk)
            stream.seek(0)
            try:
                with zipfile.ZipFile(stream) as archive:
                    members = archive.infolist()
                    names = [member.filename for member in members]
                    if not members or len(members) > MAX_MEMBERS or len(set(names)) != len(names):
                        raise ValueError("wheel member count or duplicate paths")
                    if sum(m.file_size for m in members) > MAX_EXPANDED_BYTES:
                        raise ValueError("wheel expanded size limit exceeded")
                    directory_entries = set()
                    for member in members:
                        entry = member.filename
                        is_directory = member.is_dir()
                        parts = entry[:-1].split("/") if is_directory else entry.split("/")
                        kind = stat.S_IFMT(member.external_attr >> 16)
                        if (
                            not entry
                            or entry.startswith("/")
                            or "\\" in entry
                            or any(part in ("", ".", "..") for part in parts)
                            or len(entry) > 1024
                            or kind == stat.S_IFLNK
                            or (is_directory and (member.file_size != 0 or kind not in (0, stat.S_IFDIR)))
                            or (not is_directory and kind == stat.S_IFDIR)
                        ):
                            raise ValueError("unsafe or noncanonical wheel path")
                        if is_directory:
                            directory_entries.add(entry)
                        if member.file_size > MAX_MEMBER_BYTES or member.file_size > MAX_RATIO * max(
                            1, member.compress_size
                        ):
                            raise ValueError("wheel member expansion limit exceeded")
                    candidates = [n for n in names if n.count("/") == 1 and n.endswith(".dist-info/METADATA")]
                    if len(candidates) != 1:
                        raise ValueError("wheel lacks unique METADATA")
                    distdir = candidates[0].rsplit("/", 1)[0]
                    if f"{distdir}/WHEEL" not in names or f"{distdir}/RECORD" not in names:
                        raise ValueError("wheel lacks WHEEL or RECORD metadata")
                    metadata = BytesParser().parsebytes(_wheel_member_bytes(archive, archive.getinfo(candidates[0])))
                    if (
                        canonicalize_name(metadata.get("Name", "")) != name
                        or metadata.get("Version") is None
                        or Version(metadata["Version"]) != version
                    ):
                        raise ValueError("wheel filename disagrees with METADATA name/version")
                    python_constraint = metadata.get("Requires-Python")
                    if python_constraint and not SpecifierSet(python_constraint).contains(
                        f"{sys.version_info.major}.{sys.version_info.minor}"
                    ):
                        raise ValueError("wheel Requires-Python excludes this interpreter")
                    wheel_headers = BytesParser().parsebytes(
                        _wheel_member_bytes(archive, archive.getinfo(f"{distdir}/WHEEL"))
                    )
                    if not wheel_headers.get("Wheel-Version", "").startswith("1."):
                        raise ValueError("unsupported wheel format")
                    inside_tags = set()
                    for value in wheel_headers.get_all("Tag", []):
                        inside_tags.update(parse_tag(value))
                    if not inside_tags or inside_tags != tags:
                        raise ValueError("wheel internal tags disagree with filename")
                    record = _wheel_member_bytes(archive, archive.getinfo(f"{distdir}/RECORD")).decode("utf-8")
                    rows = list(csv.reader(record.splitlines()))
                    listed = set()
                    for row in rows:
                        if len(row) != 3 or row[0] in listed or row[0] not in names or row[0] in directory_entries:
                            raise ValueError("invalid wheel RECORD")
                        listed.add(row[0])
                        if row[0] != f"{distdir}/RECORD":
                            if not row[1].startswith("sha256="):
                                raise ValueError("wheel RECORD lacks SHA-256")
                            info = archive.getinfo(row[0])
                            calculated = hashlib.sha256()
                            total = 0
                            with archive.open(info) as entry:
                                for chunk in iter(lambda: entry.read(READ_CHUNK), b""):
                                    total += len(chunk)
                                    if total > MAX_MEMBER_BYTES:
                                        raise ValueError("wheel expanded member exceeds limit")
                                    calculated.update(chunk)
                            if total != info.file_size:
                                raise ValueError("wheel RECORD member size mismatch")
                            expected = base64.urlsafe_b64encode(calculated.digest()).decode("ascii").rstrip("=")
                            if expected != row[1][7:]:
                                raise ValueError("wheel RECORD hash mismatch")
                            if row[2] and int(row[2]) != total:
                                raise ValueError("wheel RECORD size mismatch")
                    if listed != set(names) - directory_entries:
                        raise ValueError("wheel RECORD omits archive entries")
                    requires = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
            except (zipfile.BadZipFile, UnicodeError, KeyError, OSError, ValueError) as exc:
                raise ValueError(f"malformed wheel: {path.name}: {exc}") from exc
            return name, version, requires, digest.hexdigest()
    finally:
        if descriptor != -1:
            os.close(descriptor)


def create_lock(wheelhouse: Path, output: Path, *, project_root: Path = ROOT) -> Path:
    wheelhouse = Path(os.path.abspath(wheelhouse))
    output = Path(os.path.abspath(output))
    project_root = Path(os.path.abspath(project_root))
    guard_path(wheelhouse)
    guard_path(project_root)
    guard_path(output)
    if not wheelhouse.is_dir():
        raise ValueError("wheelhouse directory not found")
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise ValueError("wheelhouse has no distribution wheels")
    supported = set(sys_tags())
    versions: dict[str, Version] = {}
    graph: dict[str, list[Requirement]] = {}
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for wheel in wheels:
        name, version, dependencies, digest = _read_wheel(wheel, supported)
        if name in versions and versions[name] != version:
            raise ValueError(f"conflicting wheel versions for {name}")
        versions[name] = version
        # Two installable builds with one name/version must not imply
        # different dependency closures depending on pip's wheel choice.
        normalized = sorted(str(requirement) for requirement in dependencies)
        if name in graph and normalized != sorted(str(req) for req in graph[name]):
            raise ValueError(f"conflicting wheel dependency metadata for {name}=={version}")
        graph[name] = dependencies
        grouped[(name, str(version))].add(digest)

    active: dict[str, set[str]] = defaultdict(set)
    processed: set[tuple[str, str]] = set()

    def require(req: Requirement, *, origin: str, selected_extras: set[str]) -> None:
        # Marker environment is evaluated once for the base dependency and
        # for each explicitly requested extra of the depending distribution.
        if req.marker and not any(req.marker.evaluate({"extra": extra}) for extra in (selected_extras | {""})):
            return
        name = canonicalize_name(req.name)
        if name not in versions:
            raise ValueError(f"missing required dependency {name} (from {origin})")
        if versions[name] not in req.specifier:
            raise ValueError(f"incompatible {name}=={versions[name]} required by {origin}: {req}")
        new_extras = set(req.extras) - active[name]
        active[name].update(req.extras)
        # The empty extra represents dependencies installed unconditionally.
        for extra in {""} | new_extras:
            if (name, extra) in processed:
                continue
            processed.add((name, extra))
            for dependency in graph[name]:
                require(dependency, origin=f"{name}[{extra}]", selected_extras={extra})

    for req in required_packages(project_root):
        require(req, origin="project or validation requirements", selected_extras=set())
    # Validate all installed distributions' unconditional requirements, even
    # when an extra wheel is included in the wheelhouse but not a root.
    for name in graph:
        if (name, "") not in processed:
            require(Requirement(name), origin="wheelhouse", selected_extras=set())

    rows = [
        f"{name}=={version} " + " ".join(f"--hash=sha256:{h}" for h in sorted(digests))
        for (name, version), digests in sorted(grouped.items())
    ]
    payload = (
        "# Complete platform-specific wheelhouse inventory; not certified until a fresh\n"
        "# --no-index --require-hashes installation and import smoke test pass.\n" + "\n".join(rows) + "\n"
    ).encode("utf-8")
    guard_path(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard_path(output)
    fd, temporary = tempfile.mkstemp(prefix=".mnist-lock-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        guard_path(output)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(create_lock(args.wheelhouse, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
