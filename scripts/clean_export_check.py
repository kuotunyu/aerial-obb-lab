"""Build and inspect a release archive made only from committed files.

This gate is intentionally standard-library-only.  It does not import an ML runtime,
download artifacts, or inspect ignored files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "yolo26-dota-obb-v1.0.0rc1.zip"

REQUIRED_MEMBERS = {
    ".github/workflows/release-gates.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "README.zh-TW.md",
    "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md",
    "demo/space-static/app.js",
    "demo/space-static/index.html",
    "demo/space-static/obb.js",
    "demo/space-static/yolo26n-obb.onnx",
    "pyproject.toml",
    "release/artifact-manifest.json",
    "release/evidence.json",
    "scripts/release_check.py",
    "scripts/repo_check.py",
    "tests/fixtures/browser_parity.json",
    "tests/js/browser_parity_runner.js",
    "uv.lock",
}
PRIVATE_NAMES = {".env", "notes.private.md"}
PRIVATE_FRAGMENTS = ("interview", "面試")
RUNTIME_PREFIXES = (
    ".pytest_cache/",
    ".venv/",
    "build/",
    "datasets/",
    "dist/",
    "runs/",
    "wandb/",
)
RUNTIME_PARTS = {"__pycache__", ".ipynb_checkpoints"}
DRIVE_PATH_RE = re.compile(r"^[A-Za-z]:[/\\]")
LOCAL_PATH_BYTES_RE = re.compile(
    rb'''(?i)(?:[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:Users|home)/[^/\x00\s"']+/)'''
)
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def _normalized_member(raw: str) -> str:
    return raw.replace("\\", "/")


def _unsafe_member(raw: str) -> bool:
    normalized = _normalized_member(raw)
    path = PurePosixPath(normalized)
    return (
        not normalized
        or normalized.startswith("/")
        or bool(DRIVE_PATH_RE.match(normalized))
        or ".." in path.parts
    )


def archive_policy_errors(member_names: list[str]) -> list[str]:
    """Return deterministic path-policy violations for archive member names."""
    errors: list[str] = []
    for raw in member_names:
        normalized = _normalized_member(raw)
        if _unsafe_member(normalized):
            errors.append(f"unsafe path: {raw}")
            continue
        relative = normalized.removeprefix("./")
        lowered = relative.casefold()
        basename = PurePosixPath(relative).name.casefold()
        if basename in PRIVATE_NAMES or any(fragment.casefold() in lowered for fragment in PRIVATE_FRAGMENTS):
            errors.append(f"private path: {relative}")
        elif lowered.startswith(RUNTIME_PREFIXES) or any(part in PurePosixPath(lowered).parts for part in RUNTIME_PARTS):
            errors.append(f"runtime/model path: {relative}")
    return errors


def inspect_archive(archive: Path) -> list[str]:
    """Validate member policy, release inventory, hashes, and embedded local paths."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            infos = [info for info in bundle.infolist() if not info.is_dir()]
            names = [_normalized_member(info.filename) for info in infos]
            errors.extend(archive_policy_errors(names))
            if len(names) != len(set(names)):
                errors.append("archive contains duplicate members")
            missing = sorted(REQUIRED_MEMBERS - set(names))
            errors.extend(f"missing required member: {name}" for name in missing)
            if errors:
                return errors

            manifest = json.loads(bundle.read("release/artifact-manifest.json").decode("utf-8"))
            artifacts = manifest.get("artifacts", [])
            listed = {entry.get("path") for entry in artifacts}
            maximum = int(manifest.get("policy", {}).get("maximum_unlisted_tracked_file_bytes", 0))
            for entry in artifacts:
                name = entry.get("path", "")
                if name not in names:
                    errors.append(f"manifest artifact missing from archive: {name}")
                    continue
                payload = bundle.read(name)
                if len(payload) != entry.get("bytes"):
                    errors.append(f"{name}: byte size differs from manifest")
                if hashlib.sha256(payload).hexdigest() != entry.get("sha256"):
                    errors.append(f"{name}: SHA-256 differs from manifest")

            if maximum > 0:
                for info in infos:
                    if info.filename not in listed and info.file_size > maximum:
                        errors.append(
                            f"unlisted member exceeds {maximum} bytes: {info.filename}"
                        )

            for info in infos:
                suffix = PurePosixPath(info.filename).suffix.casefold()
                if suffix in TEXT_SUFFIXES and LOCAL_PATH_BYTES_RE.search(bundle.read(info.filename)):
                    errors.append(f"{info.filename}: absolute local user path")
            for name in listed:
                if name in names and LOCAL_PATH_BYTES_RE.search(bundle.read(name)):
                    errors.append(f"{name}: absolute local user path")
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        errors.append(f"invalid release archive: {exc}")
    return errors


def _git_output(*args: str, root: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def create_archive(output: Path, root: Path = ROOT) -> str:
    """Create a validated ZIP from a clean committed HEAD and return its SHA-256."""
    if _git_output("status", "--porcelain=v1", "--untracked-files=all", root=root):
        raise RuntimeError("working tree is not clean; commit the release candidate first")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={output}", "HEAD"],
            cwd=root,
            check=True,
        )
        errors = inspect_archive(output)
        if errors:
            raise RuntimeError("release archive failed inspection:\n  " + "\n  ".join(errors))
    except Exception:
        if output.is_file():
            output.unlink()
        raise
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inspect-only", action="store_true", help="inspect an existing --output archive"
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        if args.inspect_only:
            errors = inspect_archive(output)
            if errors:
                raise RuntimeError("release archive failed inspection:\n  " + "\n  ".join(errors))
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
        else:
            digest = create_archive(output)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] committed clean export: {output}")
    print(f"[OK] SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
