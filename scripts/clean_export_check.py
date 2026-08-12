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
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / "aerial-obb-lab-v1.0.0rc2.zip"

REQUIRED_MEMBERS = {
    ".github/workflows/release-gates.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.en.md",
    "README.md",
    "README.zh-TW.md",
    "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md",
    "demo/web/app.js",
    "demo/web/fonts/IBM-Plex-OFL.txt",
    "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
    "demo/web/index.html",
    "demo/web/obb.js",
    "demo/web/style.css",
    "docs/assets/browser-workbench.png",
    "docs/OWNER_ACTIONS.md",
    "pyproject.toml",
    "release/artifact-manifest.json",
    "release/evidence.json",
    "scripts/browser_smoke.py",
    "scripts/clean_export_check.py",
    "scripts/release_check.py",
    "scripts/repo_check.py",
    "tests/fixtures/browser-smoke.svg",
    "tests/fixtures/browser_parity.json",
    "tests/js/browser_parity_runner.js",
    "uv.lock",
}
PRIVATE_NAMES = {".env", "notes.private.md"}
PRIVATE_FRAGMENTS = ("interview", "面試")
INTERNAL_ONLY_NAMES = {"design.md", "product.md"}
INTERNAL_ONLY_PREFIXES = (".claude/", ".impeccable/", "docs/superpowers/")
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
FORBIDDEN_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".mlpackage"}
DOTA_DERIVED_VISUAL_RE = re.compile(r"^assets/hbb_vs_obb_.*\.(?:jpg|jpeg|png)$", re.I)


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
        elif basename in INTERNAL_ONLY_NAMES or lowered.startswith(INTERNAL_ONLY_PREFIXES):
            errors.append(f"internal-only path: {relative}")
        elif lowered.startswith(RUNTIME_PREFIXES) or any(part in PurePosixPath(lowered).parts for part in RUNTIME_PARTS):
            errors.append(f"runtime/model path: {relative}")
        elif PurePosixPath(lowered).suffix in FORBIDDEN_MODEL_SUFFIXES:
            errors.append(f"model binary path: {relative}")
        elif DOTA_DERIVED_VISUAL_RE.match(relative):
            errors.append(f"DOTA-derived visual path: {relative}")
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
            artifacts = manifest.get("bundled_third_party_artifacts", [])
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

            for entry in manifest.get("excluded_historical_artifacts", []):
                name = entry.get("path", "")
                if name in names:
                    errors.append(f"excluded historical artifact is still archived: {name}")

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


def _run(args: list[str], cwd: Path) -> dict[str, object]:
    started = time.monotonic()
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)
    return {"command": args, "seconds": round(time.monotonic() - started, 3), "status": "passed"}


def _distribution_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as bundle:
            return [info.filename for info in bundle.infolist() if not info.is_dir()]
    with tarfile.open(path, "r:gz") as bundle:
        return [member.name for member in bundle.getmembers() if member.isfile()]


def _distribution_errors(path: Path) -> list[str]:
    names = _distribution_members(path)
    errors = archive_policy_errors(names)
    normalized = [name.casefold().replace("\\", "/") for name in names]
    forbidden_parts = {"assets", "demo", "notebooks", "release", "tests"}
    for name in normalized:
        if forbidden_parts.intersection(PurePosixPath(name).parts):
            errors.append(f"package contains release-only member: {name}")
    if path.suffix == ".whl":
        for name in normalized:
            if not (name.startswith("obbkit/") or ".dist-info/" in name):
                errors.append(f"wheel contains unexpected member: {name}")
    return errors


def distribution_paths(folder: Path) -> list[Path]:
    """Select one wheel and one sdist while validating uv's generated ignore marker."""
    entries = sorted(folder.iterdir())
    marker = folder / ".gitignore"
    if marker in entries:
        if marker.read_text(encoding="utf-8").strip() != "*":
            raise RuntimeError("dist/.gitignore has unexpected content")
        entries.remove(marker)
    distributions = [
        path for path in entries if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    ]
    unexpected = [path.name for path in entries if path not in distributions]
    if unexpected:
        raise RuntimeError("unexpected dist members: " + ", ".join(unexpected))
    if len(distributions) != 2 or len([path for path in distributions if path.suffix == ".whl"]) != 1:
        raise RuntimeError("expected exactly one wheel and one .tar.gz source distribution")
    return sorted(distributions)


def verify_snapshot(archive: Path, run_browser: bool = True) -> dict[str, object]:
    """Extract a validated archive and rebuild all CPU/package gates from that snapshot."""
    errors = inspect_archive(archive)
    if errors:
        raise RuntimeError("release archive failed inspection:\n  " + "\n  ".join(errors))
    uv = shutil.which("uv")
    node = shutil.which("node")
    if not uv or not node:
        raise RuntimeError("uv and Node.js are required for snapshot verification")

    with tempfile.TemporaryDirectory(prefix="yolo26-obb-clean-export-") as temporary:
        temp_root = Path(temporary)
        export = temp_root / "export"
        with zipfile.ZipFile(archive) as bundle:
            committed_archive_files = len(
                [info for info in bundle.infolist() if not info.is_dir()]
            )
            symlinks = [
                info.filename
                for info in bundle.infolist()
                if ((info.external_attr >> 16) & 0o170000) == 0o120000
            ]
            if symlinks:
                raise RuntimeError("release archive contains symbolic links: " + ", ".join(symlinks))
            bundle.extractall(export)

        steps: list[dict[str, object]] = []
        steps.append(_run([uv, "sync", "--frozen", "--no-install-project"], export))
        python = export / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        steps.append(_run([str(python), "-m", "pytest", "-q"], export))
        steps.append(_run([str(python), "scripts/repo_check.py"], export))
        steps.append(_run([str(python), "scripts/release_check.py"], export))
        if run_browser:
            steps.append(
                _run(
                    [
                        str(python),
                        "scripts/browser_smoke.py",
                        "--screenshot",
                        str(temp_root / "browser-smoke.png"),
                    ],
                    export,
                )
            )
            steps.append(
                _run(
                    [
                        str(python),
                        "-c",
                        "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('ultralytics') is None",
                    ],
                    export,
                )
            )
        steps.append(_run([uv, "build"], export))

        distributions = distribution_paths(export / "dist")
        for distribution in distributions:
            package_errors = _distribution_errors(distribution)
            if package_errors:
                raise RuntimeError(
                    f"{distribution.name} failed package inspection:\n  "
                    + "\n  ".join(package_errors)
                )

        package_env = temp_root / "package-env"
        steps.append(_run([uv, "venv", "--python", "3.11", str(package_env)], export))
        package_python = package_env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        wheel = next(path for path in distributions if path.suffix == ".whl")
        steps.append(
            _run(
                [uv, "pip", "install", "--python", str(package_python), "--no-deps", str(wheel)],
                export,
            )
        )
        steps.append(
            _run(
                [
                    str(package_python),
                    "-c",
                    "import obbkit; assert obbkit.__version__ == '1.0.0rc2'; print(obbkit.__version__)",
                ],
                export,
            )
        )
        return {
            "committed_archive_files": committed_archive_files,
            "distributions": [path.name for path in distributions],
            "steps": steps,
            "result": "passed",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inspect-only", action="store_true", help="inspect an existing --output archive"
    )
    parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="skip Playwright in this job when a separate browser-smoke CI job runs it",
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        summary = None
        if args.inspect_only:
            errors = inspect_archive(output)
            if errors:
                raise RuntimeError("release archive failed inspection:\n  " + "\n  ".join(errors))
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
        else:
            digest = create_archive(output)
            summary = verify_snapshot(output, run_browser=not args.skip_browser)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] committed clean export: {output}")
    print(f"[OK] SHA-256: {digest}")
    if summary is not None:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
