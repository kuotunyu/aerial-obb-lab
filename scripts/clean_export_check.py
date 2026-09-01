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
DEFAULT_OUTPUT = ROOT / "dist" / "aerial-obb-lab-v1.0.0.zip"

REQUIRED_MEMBERS = {
    ".github/workflows/release-gates.yml",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.en.md",
    "README.md",
    "RELEASE_CHECKLIST.md",
    "THIRD_PARTY_NOTICES.md",
    "demo/web/app.js",
    "demo/web/demo-assets.js",
    "demo/web/demo-model.json",
    "demo/web/fonts/IBM-Plex-OFL.txt",
    "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
    "demo/web/index.html",
    "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
    "demo/web/obb.js",
    "demo/web/samples/boats.jpg",
    "demo/web/style.css",
    "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
    "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
    "demo/web/THIRD_PARTY_NOTICES.md",
    "docs/assets/browser-workbench.png",
    "pyproject.toml",
    "release/artifact-manifest.json",
    "release/evidence.json",
    "scripts/browser_smoke.py",
    "scripts/clean_export_check.py",
    "scripts/release_check.py",
    "scripts/repo_check.py",
    "tests/fixtures/browser_parity.json",
    "tests/js/browser_parity_runner.js",
    "uv.lock",
}
PRIVATE_NAMES = {".env", "notes.private.md"}
PRIVATE_FRAGMENTS = ("interview", "面試")
INTERNAL_ONLY_NAMES = {"design.md", "product.md"}
INTERNAL_ONLY_PREFIXES = (".claude/", ".impeccable/", "docs/superpowers/")
OBSOLETE_PUBLIC_PATHS = {
    "readme.zh-tw.md",
    "docs/owner_actions.md",
    "docs/plan.md",
    "src/obbkit/hf_checkpoint.py",
}
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
APPROVED_DEMO_MODEL = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
APPROVED_DEMO_MODEL_BYTES = 10207127
APPROVED_DEMO_MODEL_SHA256 = "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97"
SOURCE_MODEL_SHA256 = "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
SOURCE_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx"
APPROVED_MODEL_LICENSE = "AGPL-3.0-only"
APPROVED_MODEL_LICENSE_FILE = "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt"
APPROVED_MODEL_LICENSE_SHA256 = "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
APPROVED_MODEL_LICENSE_SOURCE_URL = (
    "https://raw.githubusercontent.com/ultralytics/assets/v8.4.0/LICENSE"
)
APPROVED_SANITIZATION_RECORD = (
    "demo/web/third_party/yolo26n-obb-privacy-sanitization.json"
)
APPROVED_MODIFICATION_STATUS = "metadata-only"
APPROVED_MODIFICATION_DATE = "2026-08-31"
APPROVED_MODIFIED_FIELD = "ModelProto.metadata_props[0].value"
CANONICAL_LF_ARTIFACTS = {
    "demo/web/app.js",
    "demo/web/index.html",
    "demo/web/style.css",
    APPROVED_MODEL_LICENSE_FILE,
    APPROVED_SANITIZATION_RECORD,
}
RAW_BINARY_DIGEST_MODE = "raw-binary"
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


def _approved_model_is_manifest_bound(manifest: dict | None) -> bool:
    entries = (manifest or {}).get("bundled_third_party_artifacts", [])
    matches = [entry for entry in entries if entry.get("path") == APPROVED_DEMO_MODEL]
    expected = {
        "path": APPROVED_DEMO_MODEL,
        "bytes": APPROVED_DEMO_MODEL_BYTES,
        "sha256": APPROVED_DEMO_MODEL_SHA256,
        "source_url": SOURCE_MODEL_URL,
        "source_sha256": SOURCE_MODEL_SHA256,
        "modification_status": APPROVED_MODIFICATION_STATUS,
        "modification_date": APPROVED_MODIFICATION_DATE,
        "sanitization_record": APPROVED_SANITIZATION_RECORD,
        "license": APPROVED_MODEL_LICENSE,
        "license_file": APPROVED_MODEL_LICENSE_FILE,
    }
    return (
        len(matches) == 1
        and all(matches[0].get(field) == value for field, value in expected.items())
        and (
            "digest_mode" not in matches[0]
            or matches[0].get("digest_mode") == RAW_BINARY_DIGEST_MODE
        )
    )


def _artifact_digest_mode_error(entry: dict) -> str | None:
    relative = str(entry.get("path", ""))
    mode = entry.get("digest_mode")
    if relative in CANONICAL_LF_ARTIFACTS:
        if mode != "canonical-lf":
            return f"{relative}: canonical text artifact digest_mode must be canonical-lf"
        return None
    if "digest_mode" in entry and mode != RAW_BINARY_DIGEST_MODE:
        return (
            f"{relative}: binary artifact digest_mode must be absent or raw-binary"
        )
    return None


def _approved_license_is_manifest_bound(manifest: dict | None) -> bool:
    entries = (manifest or {}).get("bundled_third_party_artifacts", [])
    matches = [
        entry for entry in entries if entry.get("path") == APPROVED_MODEL_LICENSE_FILE
    ]
    expected = {
        "path": APPROVED_MODEL_LICENSE_FILE,
        "bytes": 34523,
        "sha256": APPROVED_MODEL_LICENSE_SHA256,
        "digest_mode": "canonical-lf",
        "source_url": APPROVED_MODEL_LICENSE_SOURCE_URL,
        "license": APPROVED_MODEL_LICENSE,
    }
    return len(matches) == 1 and all(
        matches[0].get(field) == value for field, value in expected.items()
    )


def _archive_demo_contract_errors(
    bundle: zipfile.ZipFile, manifest: dict
) -> list[str]:
    """Bind the approved exception to its same-archive manifest and receipt."""
    demo = json.loads(bundle.read("demo/web/demo-model.json").decode("utf-8"))
    receipt = json.loads(bundle.read(APPROVED_SANITIZATION_RECORD).decode("utf-8"))
    errors: list[str] = []
    if not _approved_license_is_manifest_bound(manifest):
        errors.append(
            f"{APPROVED_MODEL_LICENSE_FILE}: canonical-LF license identity is not exact"
        )
    expected_relative = APPROVED_DEMO_MODEL.removeprefix("demo/web/")
    if (
        demo.get("model", {}).get("path"),
        demo.get("model", {}).get("bytes"),
        demo.get("model", {}).get("sha256"),
        demo.get("model", {}).get("source"),
        demo.get("model", {}).get("sourceSha256"),
        demo.get("model", {}).get("modificationStatus"),
        demo.get("model", {}).get("license"),
        demo.get("model", {}).get("release"),
    ) != (
        expected_relative,
        APPROVED_DEMO_MODEL_BYTES,
        APPROVED_DEMO_MODEL_SHA256,
        SOURCE_MODEL_URL,
        SOURCE_MODEL_SHA256,
        APPROVED_MODIFICATION_STATUS,
        APPROVED_MODEL_LICENSE,
        "v8.4.0",
    ):
        errors.append("demo-model.json: approved derivative contract differs")
    if (
        demo.get("license", {}).get("path"),
        demo.get("license", {}).get("sha256"),
        demo.get("sanitization", {}).get("path"),
        demo.get("sanitization", {}).get("modificationDate"),
        demo.get("sanitization", {}).get("modifiedField"),
        demo.get("sanitization", {}).get("removedMetadataEntries"),
    ) != (
        APPROVED_MODEL_LICENSE_FILE.removeprefix("demo/web/"),
        APPROVED_MODEL_LICENSE_SHA256,
        APPROVED_SANITIZATION_RECORD.removeprefix("demo/web/"),
        APPROVED_MODIFICATION_DATE,
        APPROVED_MODIFIED_FIELD,
        1,
    ):
        errors.append("demo-model.json: license/sanitization contract differs")
    if (
        receipt.get("derivative", {}).get("path"),
        receipt.get("derivative", {}).get("bytes"),
        receipt.get("derivative", {}).get("sha256"),
        receipt.get("source", {}).get("url"),
        receipt.get("source", {}).get("sha256"),
        receipt.get("license", {}).get("path"),
        receipt.get("license", {}).get("sha256"),
        receipt.get("license", {}).get("spdx"),
    ) != (
        expected_relative,
        APPROVED_DEMO_MODEL_BYTES,
        APPROVED_DEMO_MODEL_SHA256,
        SOURCE_MODEL_URL,
        SOURCE_MODEL_SHA256,
        APPROVED_MODEL_LICENSE_FILE.removeprefix("demo/web/"),
        APPROVED_MODEL_LICENSE_SHA256,
        APPROVED_MODEL_LICENSE,
    ):
        errors.append("sanitization receipt: approved identity differs")
    if (
        receipt.get("transformation", {}).get("modificationStatus"),
        receipt.get("transformation", {}).get("modificationDate"),
        receipt.get("transformation", {}).get("modifiedField"),
        receipt.get("transformation", {}).get("removedMetadataEntries"),
    ) != (
        APPROVED_MODIFICATION_STATUS,
        APPROVED_MODIFICATION_DATE,
        APPROVED_MODIFIED_FIELD,
        1,
    ):
        errors.append("sanitization receipt: modification record differs")
    return errors


def archive_policy_errors(
    member_names: list[str], manifest: dict | None = None
) -> list[str]:
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
        if lowered in OBSOLETE_PUBLIC_PATHS:
            errors.append(f"obsolete public surface: {relative}")
        elif basename in PRIVATE_NAMES or any(fragment.casefold() in lowered for fragment in PRIVATE_FRAGMENTS):
            errors.append(f"private path: {relative}")
        elif basename in INTERNAL_ONLY_NAMES or lowered.startswith(INTERNAL_ONLY_PREFIXES):
            errors.append(f"internal-only path: {relative}")
        elif lowered.startswith(RUNTIME_PREFIXES) or any(part in PurePosixPath(lowered).parts for part in RUNTIME_PARTS):
            errors.append(f"runtime/model path: {relative}")
        elif (
            PurePosixPath(lowered).suffix in FORBIDDEN_MODEL_SUFFIXES
            and not (
                relative == APPROVED_DEMO_MODEL
                and _approved_model_is_manifest_bound(manifest)
            )
        ):
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
            if len(names) != len(set(names)):
                errors.append("archive contains duplicate members")
            missing = sorted(REQUIRED_MEMBERS - set(names))
            errors.extend(f"missing required member: {name}" for name in missing)
            if errors:
                return errors

            manifest = json.loads(bundle.read("release/artifact-manifest.json").decode("utf-8"))
            errors.extend(archive_policy_errors(names, manifest))
            errors.extend(_archive_demo_contract_errors(bundle, manifest))
            artifacts = manifest.get("bundled_third_party_artifacts", [])
            listed = {entry.get("path") for entry in artifacts}
            maximum = int(manifest.get("policy", {}).get("maximum_unlisted_tracked_file_bytes", 0))
            for entry in artifacts + manifest.get("reviewed_public_artifacts", []):
                name = entry.get("path", "")
                digest_mode_error = _artifact_digest_mode_error(entry)
                if digest_mode_error:
                    errors.append(digest_mode_error)
                    continue
                if name not in names:
                    errors.append(f"reviewed artifact missing from archive: {name}")
                    continue
                payload = bundle.read(name)
                if entry.get("digest_mode") == "canonical-lf":
                    payload = payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
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
                        "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('ultralytics') is None; assert importlib.util.find_spec('huggingface_hub') is None",
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
                    "import obbkit; assert obbkit.__version__ == '1.0.0'; print(obbkit.__version__)",
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
