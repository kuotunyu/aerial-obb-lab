from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from scripts.clean_export_check import (
    DEFAULT_OUTPUT,
    REQUIRED_MEMBERS,
    archive_policy_errors,
    distribution_paths,
    inspect_archive,
    main,
    verify_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _approved_manifest() -> dict:
    return {
        "bundled_third_party_artifacts": [
            {
                "path": "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
                "bytes": 10207127,
                "sha256": "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
                "source_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx",
                "source_sha256": "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38",
                "modification_status": "metadata-only",
                "modification_date": "2026-08-31",
                "sanitization_record": "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
                "license": "AGPL-3.0-only",
                "license_file": "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
            }
        ]
    }


def _committed_candidate_archive(
    tmp_path: Path, source_root: Path = ROOT
) -> Path:
    if not (source_root / ".git").exists():
        archive = tmp_path / "candidate.zip"
        ignored_parts = {
            ".git",
            ".venv",
            ".pytest_cache",
            ".superpowers",
            "__pycache__",
            "build",
            "dist",
        }
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as bundle:
            for source in sorted(source_root.rglob("*")):
                relative = source.relative_to(source_root)
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or ignored_parts.intersection(relative.parts)
                ):
                    continue
                info = zipfile.ZipInfo(
                    relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                bundle.writestr(info, source.read_bytes())
        return archive

    repository = tmp_path / "candidate"
    repository.mkdir()
    base_archive = tmp_path / "base.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={base_archive}", "HEAD"],
        cwd=source_root,
        check=True,
    )
    with zipfile.ZipFile(base_archive) as bundle:
        bundle.extractall(repository)

    modified = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only", "-z"],
        cwd=source_root,
        text=False,
    ).split(b"\0")
    for raw in modified:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        source = source_root / relative
        destination = repository / relative
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        elif destination.exists():
            destination.unlink()

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Release Test"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "-f", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "candidate"], cwd=repository, check=True)
    archive = tmp_path / "candidate.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={archive}", "HEAD"],
        cwd=repository,
        check=True,
    )
    return archive


def _repo_external_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshot"
    if (ROOT / ".git").exists():
        source = tmp_path / "source.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={source}", "HEAD"],
            cwd=ROOT,
            check=True,
        )
        with zipfile.ZipFile(source) as bundle:
            bundle.extractall(snapshot)
    else:
        shutil.copytree(
            ROOT,
            snapshot,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                ".pytest_cache",
                ".superpowers",
                "__pycache__",
                "build",
                "dist",
            ),
        )
    assert not (snapshot / ".git").exists()
    return snapshot


def _mutated_git_archive(tmp_path: Path, field: str, value: str) -> Path:
    source = _committed_candidate_archive(tmp_path)
    target = tmp_path / "mutated.zip"
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(target, "w") as outgoing:
        for info in incoming.infolist():
            payload = incoming.read(info.filename)
            if info.filename == "release/artifact-manifest.json":
                manifest = json.loads(payload.decode("utf-8"))
                model = next(
                    item
                    for item in manifest["bundled_third_party_artifacts"]
                    if item["path"]
                    == "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
                )
                model[field] = value
                payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            outgoing.writestr(info, payload)
    return target


def _mutated_license_archive(
    tmp_path: Path, field: str, value: str | None
) -> Path:
    source = _committed_candidate_archive(tmp_path)
    target = tmp_path / "mutated-license.zip"
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(target, "w") as outgoing:
        for info in incoming.infolist():
            payload = incoming.read(info.filename)
            if info.filename == "release/artifact-manifest.json":
                manifest = json.loads(payload.decode("utf-8"))
                license_entry = next(
                    item
                    for item in manifest["bundled_third_party_artifacts"]
                    if item["path"]
                    == "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt"
                )
                if value is None:
                    license_entry.pop(field, None)
                else:
                    license_entry[field] = value
                payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            outgoing.writestr(info, payload)
    return target


def _mutated_binary_digest_mode_archive(
    tmp_path: Path, artifact_path: str, digest_mode: str | None
) -> Path:
    source = _committed_candidate_archive(tmp_path)
    target = tmp_path / "mutated-binary-mode.zip"
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(target, "w") as outgoing:
        for info in incoming.infolist():
            payload = incoming.read(info.filename)
            if info.filename == "release/artifact-manifest.json":
                manifest = json.loads(payload.decode("utf-8"))
                artifact = next(
                    item
                    for item in manifest["bundled_third_party_artifacts"]
                    if item["path"] == artifact_path
                )
                artifact["digest_mode"] = digest_mode
                payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            outgoing.writestr(info, payload)
    return target


def test_archive_policy_rejects_private_and_runtime_paths() -> None:
    assert archive_policy_errors(
        ["README.md", "notes.private.md", "runs/x/best.pt", "datasets/DOTAv1/a.png"]
    ) == [
        "private path: notes.private.md",
        "runtime/model path: runs/x/best.pt",
        "runtime/model path: datasets/DOTAv1/a.png",
    ]


def test_archive_policy_rejects_internal_design_tool_files() -> None:
    assert archive_policy_errors(
        [
            "PRODUCT.md",
            "DESIGN.md",
            ".claude/launch.json",
            ".impeccable/design.json",
            "docs/superpowers/plans/private-plan.md",
        ]
    ) == [
        "internal-only path: PRODUCT.md",
        "internal-only path: DESIGN.md",
        "internal-only path: .claude/launch.json",
        "internal-only path: .impeccable/design.json",
        "internal-only path: docs/superpowers/plans/private-plan.md",
    ]


def test_archive_policy_rejects_obsolete_operational_surfaces() -> None:
    retired = [
        "README.zh-TW.md",
        "docs/OWNER_ACTIONS.md",
        "docs/PLAN.md",
        "src/obbkit/hf_checkpoint.py",
    ]

    assert archive_policy_errors(retired) == [
        f"obsolete public surface: {relative}" for relative in retired
    ]


def test_archive_policy_rejects_unsafe_tar_members() -> None:
    fake_home_path = "C:/" + "Users/alice/file.txt"
    assert archive_policy_errors(["../escape.txt", "/absolute.txt", fake_home_path]) == [
        "unsafe path: ../escape.txt",
        "unsafe path: /absolute.txt",
        f"unsafe path: {fake_home_path}",
    ]


def test_clean_export_has_a_snapshot_rebuild_gate() -> None:
    assert callable(verify_snapshot)
    assert "run_browser" in inspect.signature(verify_snapshot).parameters


def test_archive_policy_rejects_unapproved_model_and_dota_visuals() -> None:
    assert archive_policy_errors(
        [
            "README.md",
            "demo/web/yolo26n-obb.onnx",
            "assets/hbb_vs_obb_1_P0706_ship.jpg",
        ]
    ) == [
        "model binary path: demo/web/yolo26n-obb.onnx",
        "DOTA-derived visual path: assets/hbb_vs_obb_1_P0706_ship.jpg",
    ]


def test_clean_export_admits_only_the_reviewed_derivative_demo_model() -> None:
    approved_demo_model = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
    manifest = _approved_manifest()

    assert archive_policy_errors(["README.md", approved_demo_model], manifest) == []
    assert archive_policy_errors(
        ["README.md", approved_demo_model, "demo/web/models/second.onnx"],
        manifest,
    ) == ["model binary path: demo/web/models/second.onnx"]
    assert archive_policy_errors(["README.md", approved_demo_model], {}) == [
        f"model binary path: {approved_demo_model}"
    ]


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("license", "MIT"),
        ("license_file", "demo/web/third_party/OTHER.txt"),
        ("sanitization_record", "demo/web/third_party/other.json"),
        ("modification_date", "2026-09-01"),
        ("source_url", "https://example.invalid/model.onnx"),
        ("source_sha256", "0" * 64),
        ("license", None),
        ("source_url", None),
    ],
)
def test_clean_export_rejects_derivative_identity_mutation(
    field: str, mutation: str | None
) -> None:
    approved_demo_model = "demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
    manifest = _approved_manifest()
    if mutation is None:
        manifest["bundled_third_party_artifacts"][0].pop(field)
    else:
        manifest["bundled_third_party_artifacts"][0][field] = mutation

    assert archive_policy_errors([approved_demo_model], manifest) == [
        f"model binary path: {approved_demo_model}"
    ]


def test_inspect_only_rejects_mutated_derivative_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _mutated_git_archive(tmp_path, "license", "MIT")

    assert (
        "model binary path: demo/web/models/yolo26n-obb-privacy-sanitized.onnx"
        in inspect_archive(archive)
    )
    assert main(["--inspect-only", "--output", str(archive)]) == 1
    assert "model binary path" in capsys.readouterr().err


def test_pristine_committed_archive_passes_inspect_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _committed_candidate_archive(tmp_path)

    assert inspect_archive(archive) == []
    assert main(["--inspect-only", "--output", str(archive)]) == 0
    assert "[OK] committed clean export" in capsys.readouterr().out


def test_pristine_exported_snapshot_passes_inspect_only_without_git_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = _repo_external_snapshot(tmp_path)
    archive_root = tmp_path / "archive-build"
    archive_root.mkdir()

    archive = _committed_candidate_archive(archive_root, source_root=snapshot)

    assert inspect_archive(archive) == []
    assert main(["--inspect-only", "--output", str(archive)]) == 0
    assert "[OK] committed clean export" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("digest_mode", None),
        ("digest_mode", "binary"),
        ("sha256", "0" * 64),
    ],
)
def test_inspect_only_rejects_license_digest_contract_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    mutation: str | None,
) -> None:
    archive = _mutated_license_archive(tmp_path, field, mutation)

    errors = inspect_archive(archive)
    assert any(
        error.startswith(
            "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt: "
        )
        for error in errors
    )
    assert main(["--inspect-only", "--output", str(archive)]) == 1
    assert "ULTRALYTICS-AGPL-3.0.txt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("artifact_path", "digest_mode"),
    [
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            "canonical-lf",
        ),
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            "unexpected-text-mode",
        ),
        (
            "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
            None,
        ),
        ("demo/web/samples/boats.jpg", "canonical-lf"),
    ],
)
def test_inspect_only_rejects_binary_digest_mode_misuse(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    artifact_path: str,
    digest_mode: str | None,
) -> None:
    archive = _mutated_binary_digest_mode_archive(
        tmp_path, artifact_path, digest_mode
    )
    expected = (
        f"{artifact_path}: binary artifact digest_mode must be absent or raw-binary"
    )

    errors = inspect_archive(archive)
    assert expected in errors
    assert not any(error.startswith("invalid release archive:") for error in errors)
    assert main(["--inspect-only", "--output", str(archive)]) == 1
    assert expected in capsys.readouterr().err


def test_inspect_only_accepts_explicit_raw_binary_model_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _mutated_binary_digest_mode_archive(
        tmp_path,
        "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
        "raw-binary",
    )

    assert inspect_archive(archive) == []
    assert main(["--inspect-only", "--output", str(archive)]) == 0
    assert "[OK] committed clean export" in capsys.readouterr().out


def test_clean_export_keeps_its_own_gate_and_real_demo_assets() -> None:
    assert {
        "README.en.md",
        "demo/web/app.js",
        "demo/web/fonts/IBM-Plex-OFL.txt",
        "demo/web/fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "demo/web/index.html",
        "demo/web/obb.js",
        "demo/web/style.css",
        "docs/assets/browser-workbench.png",
        "scripts/clean_export_check.py",
        "scripts/browser_smoke.py",
        "demo/web/demo-assets.js",
        "demo/web/demo-model.json",
        "demo/web/models/yolo26n-obb-privacy-sanitized.onnx",
        "demo/web/samples/boats.jpg",
        "demo/web/third_party/ULTRALYTICS-AGPL-3.0.txt",
        "demo/web/third_party/yolo26n-obb-privacy-sanitization.json",
    } <= REQUIRED_MEMBERS
    assert not any("gradio" in member.casefold() for member in REQUIRED_MEMBERS)


def test_release_archive_keeps_real_demo_assets_and_omits_internal_docs(
    tmp_path: Path,
) -> None:
    attributes = ROOT / ".gitattributes"
    assert attributes.is_file(), "release archive attributes are missing"

    repository = tmp_path / "repository"
    repository.mkdir()
    members = {
        ".gitattributes": attributes.read_bytes(),
        "README.md": b"public release\n",
        "demo/web/samples/boats.jpg": b"reviewed image placeholder\n",
        "docs/superpowers/plans/2026-08-31-aerial-obb-pages-showcase.md": b"internal plan\n",
        "docs/superpowers/specs/2026-08-31-aerial-obb-pages-showcase-design.md": b"internal spec\n",
    }
    for relative, payload in members.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Release Test"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "release-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    archive = tmp_path / "release.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={archive}", "HEAD"],
        cwd=repository,
        check=True,
    )

    with zipfile.ZipFile(archive) as bundle:
        archived = {info.filename for info in bundle.infolist() if not info.is_dir()}

    assert "demo/web/samples/boats.jpg" in archived
    assert not any(name.startswith("docs/superpowers/") for name in archived)


def test_clean_export_omits_obsolete_operational_surfaces() -> None:
    for retired in (
        "README.zh-TW.md",
        "docs/OWNER_ACTIONS.md",
        "docs/PLAN.md",
        "src/obbkit/hf_checkpoint.py",
    ):
        assert retired not in REQUIRED_MEMBERS


def test_clean_export_default_is_stable_v1() -> None:
    assert DEFAULT_OUTPUT.name == "aerial-obb-lab-v1.0.0.zip"


def test_distribution_selector_accepts_uv_build_marker_only(tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("*", encoding="utf-8")
    (tmp_path / "package-1.0.tar.gz").write_bytes(b"sdist")
    (tmp_path / "package-1.0-py3-none-any.whl").write_bytes(b"wheel")

    assert [path.name for path in distribution_paths(tmp_path)] == [
        "package-1.0-py3-none-any.whl",
        "package-1.0.tar.gz",
    ]
