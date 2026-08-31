from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import scripts.pages_artifact_check as pages_check
from scripts.pages_artifact_check import verify_pages_tree


ROOT = Path(__file__).resolve().parents[1]
PAGES_TREE = ROOT / "demo" / "web"


def copied_pages_tree(tmp_path: Path) -> Path:
    site = tmp_path / "web"
    shutil.copytree(PAGES_TREE, site)
    return site


def joined_errors(site: Path) -> str:
    return "\n".join(verify_pages_tree(site))


def test_current_pages_tree_passes() -> None:
    assert verify_pages_tree(PAGES_TREE) == []


def test_pages_tree_admits_exact_real_demo_inventory(
    tmp_path: Path,
) -> None:
    site = copied_pages_tree(tmp_path)

    assert verify_pages_tree(site) == []
    assert (site / "samples" / "boats.jpg").is_file()
    assert (site / "models" / "yolo26n-obb-privacy-sanitized.onnx").is_file()
    assert (site / "demo-model.json").is_file()
    assert (site / "demo-assets.js").is_file()
    assert not (site / "showcase-fixture.js").exists()
    assert not (site / "fixtures" / "showcase.svg").exists()
    assert (
        site / "third_party" / "yolo26n-obb-privacy-sanitization.json"
    ).is_file()


def test_pages_tree_rejects_upstream_model_digest_alternate_model_and_binary_path_leak(
    tmp_path: Path,
) -> None:
    upstream = copied_pages_tree(tmp_path / "upstream")
    model = upstream / "models" / "yolo26n-obb-privacy-sanitized.onnx"
    model.write_bytes(bytes.fromhex("02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"))
    assert "published model bytes differ" in joined_errors(upstream)
    source_errors: list[str] = []
    pages_check._check_model_binary(
        b"model-bytes-are-hashed-by-the-caller",
        pages_check.SOURCE_MODEL_SHA256,
        source_errors,
    )
    assert source_errors == [
        "models/yolo26n-obb-privacy-sanitized.onnx: upstream model bytes are forbidden"
    ]

    alternate = copied_pages_tree(tmp_path / "alternate")
    (alternate / "models" / "alternate.onnx").write_bytes(b"alternate")
    assert "models/alternate.onnx: forbidden model/runtime artifact" in joined_errors(
        alternate
    )

    leaked = copied_pages_tree(tmp_path / "leaked")
    leaked_model = leaked / "models" / "yolo26n-obb-privacy-sanitized.onnx"
    leaked_model.write_bytes(leaked_model.read_bytes() + b"C:" + b"\\Users\\public-leak")
    assert "binary contains absolute user path" in joined_errors(leaked)


def test_pages_tree_rejects_model_dota_secret_path_and_origin(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    (site / "model.onnx").write_bytes(b"x")
    (site / "dota-derived.png").write_bytes(b"x")
    (site / "leak.js").write_text(
        'const t="ghp_' + 'x' * 24 + '";'
        'const p="C:' + '\\\\Users\\\\alice\\\\private.onnx";'
        'const u="https://unapproved.example/runtime.js";',
        encoding="utf-8",
    )

    joined = joined_errors(site)

    assert "forbidden model/runtime artifact" in joined
    assert "forbidden DOTA-derived path" in joined
    assert "token-shaped string" in joined
    assert "absolute user path" in joined
    assert "unapproved external origin" in joined


def test_pages_tree_rejects_symlinks(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    link = site / "linked.js"
    try:
        link.symlink_to(site / "app.js")
    except OSError as exc:
        if sys.platform != "win32":
            pytest.skip(f"symbolic links are unavailable: {exc}")
        target = tmp_path / "junction-target"
        target.mkdir()
        link = site / "linked"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"symbolic links and junctions are unavailable: {exc}")

    assert f"{link.name}: symbolic link" in verify_pages_tree(site)


@pytest.mark.parametrize(
    "relative",
    (
        "index.html",
        "app.js",
        "obb.js",
        "demo-assets.js",
        "style.css",
        "fonts/IBMPlexSansCondensed-SemiBold.woff2",
        "fonts/IBM-Plex-OFL.txt",
        "README.md",
        "samples/boats.jpg",
        "models/yolo26n-obb-privacy-sanitized.onnx",
        "demo-model.json",
        "third_party/ULTRALYTICS-AGPL-3.0.txt",
        "third_party/yolo26n-obb-privacy-sanitization.json",
        "THIRD_PARTY_NOTICES.md",
    ),
)
def test_pages_tree_rejects_required_file_absence(tmp_path: Path, relative: str) -> None:
    site = copied_pages_tree(tmp_path)
    (site / relative).unlink()

    assert f"{relative}: required Pages file is missing" in verify_pages_tree(site)


def test_pages_tree_rejects_unexpected_binary(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    (site / "payload.bin").write_bytes(b"\x00\x01\x02")

    assert "payload.bin: unexpected binary file" in verify_pages_tree(site)


def test_pages_tree_rejects_hard_links(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    alias = site / "app-alias.js"
    try:
        os.link(site / "app.js", alias)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    errors = verify_pages_tree(site)

    assert any(error.endswith(": hard link count is 2") for error in errors)


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    (
        (
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js",
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.2/dist/ort.min.js",
            "exact ORT script URL is missing",
        ),
        (
            "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp",
            "sha384-invalid",
            "exact ORT integrity is missing",
        ),
        (
            'script.crossOrigin = "anonymous"',
            'script.crossOrigin = "use-credentials"',
            "exact ORT anonymous CORS setting is missing",
        ),
    ),
)
def test_pages_tree_rejects_exact_ort_tuple_mismatch(
    tmp_path: Path, old: str, new: str, reason: str
) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(app.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    assert f"app.js: {reason}" in verify_pages_tree(site)


def test_pages_tree_rejects_oversized_text_but_allows_reviewed_font(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    (site / "oversized.js").write_text("x" * (1024 * 1024 + 1), encoding="utf-8")

    joined = joined_errors(site)

    assert "oversized.js: file exceeds 1 MiB" in joined
    assert "fonts/IBMPlexSansCondensed-SemiBold.woff2: file exceeds 1 MiB" not in joined


def test_pages_tree_scans_documentation_but_does_not_treat_its_urls_as_requests(
    tmp_path: Path,
) -> None:
    site = copied_pages_tree(tmp_path)
    (site / "notes.md").write_text(
        "Reference: https://unapproved.example/docs\n"
        "Do not paste ghp_" + "x" * 24 + " here.\n",
        encoding="utf-8",
    )

    joined = joined_errors(site)

    assert "notes.md: token-shaped string" in joined
    assert "notes.md: unapproved external origin" not in joined


def test_pages_tree_allows_reviewed_github_navigation_only_in_html(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(
        app.read_text(encoding="utf-8")
        + '\nconst executable = "https://github.com/kuotunyu/runtime.js";\n',
        encoding="utf-8",
    )

    joined = joined_errors(site)

    assert "index.html: unapproved external origin" not in joined
    assert "app.js: unapproved external origin" in joined


@pytest.mark.parametrize(
    ("relative", "old", "new", "reason"),
    (
        (
            "demo-assets.js",
            'path: "models/yolo26n-obb-privacy-sanitized.onnx"',
            'path: "models/alternate.onnx"',
            "exact derivative model path is missing",
        ),
        (
            "style.css",
            'url("fonts/IBMPlexSansCondensed-SemiBold.woff2")',
            'url("fonts/other.woff2")',
            "exact reviewed font reference is missing",
        ),
    ),
)
def test_pages_tree_rejects_required_reference_mismatch(
    tmp_path: Path, relative: str, old: str, new: str, reason: str
) -> None:
    site = copied_pages_tree(tmp_path)
    path = site / relative
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    assert f"{relative}: {reason}" in verify_pages_tree(site)


@pytest.mark.parametrize(
    ("relative", "reason"),
    (
        ("demo-assets.js", "reviewed demo asset loader bytes differ"),
        (
            "fonts/IBMPlexSansCondensed-SemiBold.woff2",
            "reviewed font bytes differ",
        ),
        ("fonts/IBM-Plex-OFL.txt", "reviewed font license bytes differ"),
        ("samples/boats.jpg", "reviewed sample image bytes differ"),
        (
            "models/yolo26n-obb-privacy-sanitized.onnx",
            "published model bytes differ",
        ),
        ("demo-model.json", "reviewed demo manifest bytes differ"),
        (
            "third_party/ULTRALYTICS-AGPL-3.0.txt",
            "reviewed Ultralytics license bytes differ",
        ),
        (
            "third_party/yolo26n-obb-privacy-sanitization.json",
            "reviewed sanitization record bytes differ",
        ),
        ("THIRD_PARTY_NOTICES.md", "reviewed third-party notice bytes differ"),
    ),
)
def test_pages_tree_rejects_reviewed_asset_content_mismatch(
    tmp_path: Path, relative: str, reason: str
) -> None:
    site = copied_pages_tree(tmp_path)
    path = site / relative
    path.write_bytes(path.read_bytes() + b"\n")

    assert f"{relative}: {reason}" in verify_pages_tree(site)


def test_final_pages_tree_rejects_synthetic_paths_and_references(tmp_path: Path) -> None:
    path_site = copied_pages_tree(tmp_path / "path")
    (path_site / "showcase-fixture.js").write_text("fixture", encoding="utf-8")
    assert "showcase-fixture.js: unexpected Pages file" in joined_errors(path_site)

    reference_site = copied_pages_tree(tmp_path / "reference")
    app = reference_site / "app.js"
    app.write_text(app.read_text(encoding="utf-8") + "\n// Synthetic legacy\n", encoding="utf-8")
    assert "app.js: current Synthetic reference is forbidden" in joined_errors(
        reference_site
    )


def test_pages_tree_rejects_synthetic_reference_in_markdown(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    readme = site / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + "\nLegacy Synthetic mode.\n",
        encoding="utf-8",
    )

    assert "README.md: current Synthetic reference is forbidden" in joined_errors(site)


def test_pages_tree_scans_macos_and_linux_home_paths(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    macos_home = "/" + "Users/alice/private.onnx"
    linux_home = "/" + "home/bob/private.onnx"
    (site / "paths.txt").write_text(
        f"{macos_home}\n{linux_home}\n",
        encoding="utf-8",
    )

    errors = verify_pages_tree(site)

    assert "paths.txt: absolute user path" in errors


@pytest.mark.parametrize(
    ("relative", "payload"),
    (
        ("extra.html", "<p>extra executable surface</p>"),
        ("extra.js", "const extra = true;"),
        ("labels.json", '{"objects":[[91,72,18,9,0.42]]}'),
        ("derived.svg", '<svg><polygon points="1,2 3,4 5,6"/></svg>'),
        ("extra.css", "body { background: red; }"),
    ),
)
def test_pages_tree_rejects_unreviewed_runtime_capable_files(
    tmp_path: Path, relative: str, payload: str
) -> None:
    site = copied_pages_tree(tmp_path)
    (site / relative).write_text(payload, encoding="utf-8")

    assert f"{relative}: unexpected Pages file" in verify_pages_tree(site)


def test_pages_tree_rejects_disguised_readme_content_substitution(
    tmp_path: Path,
) -> None:
    site = copied_pages_tree(tmp_path)
    (site / "README.md").write_text(
        "# Geometry notes\n\n"
        "sample = [[91, 72, 18, 9, 0.42], [144, 83, 20, 7, 1.17]]\n",
        encoding="utf-8",
    )

    assert "README.md: reviewed README bytes differ" in verify_pages_tree(site)


@pytest.mark.parametrize("newline", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_pages_tree_accepts_reviewed_readme_across_line_endings(
    tmp_path: Path, newline: str
) -> None:
    site = copied_pages_tree(tmp_path)
    readme = site / "README.md"
    canonical_text = readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    readme.write_bytes(canonical_text.replace("\n", newline).encode("utf-8"))

    assert verify_pages_tree(site) == []


def test_pages_tree_accepts_archive_line_endings_for_all_reviewed_text(
    tmp_path: Path,
) -> None:
    site = copied_pages_tree(tmp_path)
    for path in site.rglob("*"):
        if path.is_file() and path.suffix.casefold() in {
            ".css",
            ".html",
            ".js",
            ".md",
            ".svg",
            ".txt",
        }:
            canonical = (
                path.read_bytes()
                .decode("utf-8")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
            path.write_bytes(canonical.replace("\n", "\r\n").encode("utf-8"))

    assert verify_pages_tree(site) == []


@pytest.mark.parametrize(
    "payload",
    (
        'localStorage.setItem("model", "cached");',
        'indexedDB.open("private-models");',
        'navigator.sendBeacon("/telemetry", "used");',
    ),
)
def test_pages_tree_rejects_storage_and_telemetry_apis(
    tmp_path: Path, payload: str
) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(app.read_text(encoding="utf-8") + "\n" + payload, encoding="utf-8")

    assert "app.js: forbidden browser storage/network API" in verify_pages_tree(site)


def test_pages_tree_rejects_protocol_relative_resource_origin(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    html = site / "index.html"
    html.write_text(
        html.read_text(encoding="utf-8")
        + '\n<script src="//unapproved.example/runtime.js"></script>\n',
        encoding="utf-8",
    )

    assert "index.html: unapproved external origin" in joined_errors(site)


def test_pages_tree_rejects_relative_remote_model_fallback(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(
        app.read_text(encoding="utf-8") + '\nfetch("/model.onnx");\n',
        encoding="utf-8",
    )

    assert "app.js: unapproved runtime/model reference" in verify_pages_tree(site)


def test_pages_tree_does_not_reuse_anchor_approval_for_resource_url(
    tmp_path: Path,
) -> None:
    site = copied_pages_tree(tmp_path)
    html = site / "index.html"
    github_url = "https://github.com/kuotunyu/aerial-obb-lab"
    html.write_text(
        html.read_text(encoding="utf-8") + f'\n<img src="{github_url}" alt="bad">\n',
        encoding="utf-8",
    )

    assert "index.html: unapproved external origin" in joined_errors(site)


@pytest.mark.parametrize("comment_prefix", ("// ", "/*\n"))
def test_pages_tree_rejects_inert_comment_copy_of_ort_assignment(
    tmp_path: Path, comment_prefix: str
) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    source = app.read_text(encoding="utf-8")
    replacement = comment_prefix + "script.src = ORT_URL;"
    if comment_prefix.startswith("/*"):
        replacement += "\n*/"
    app.write_text(source.replace("script.src = ORT_URL;", replacement), encoding="utf-8")

    assert "app.js: effective dynamic ORT source is not exact" in verify_pages_tree(site)


@pytest.mark.parametrize(
    "override",
    (
        'script.src = "/local-runtime.js";',
        'script.integrity = "sha384-later-override";',
        'script.crossOrigin = "use-credentials";',
        'globalThis.ort.env.wasm.wasmPaths = "/local-wasm/";',
    ),
)
def test_pages_tree_rejects_later_ort_assignment_overrides(
    tmp_path: Path, override: str
) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(app.read_text(encoding="utf-8") + "\n" + override, encoding="utf-8")

    assert "app.js: effective ORT tuple is overridden" in verify_pages_tree(site)


def test_pages_tree_rejects_exact_wasm_base_mutation(tmp_path: Path) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(
        app.read_text(encoding="utf-8").replace(
            'const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/'
            'onnxruntime-web@1.20.1/dist/";',
            'const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/'
            'onnxruntime-web@1.20.1/wasm/";',
        ),
        encoding="utf-8",
    )

    assert "app.js: exact ORT WASM base URL is missing" in verify_pages_tree(site)


@pytest.mark.parametrize(
    "bypass",
    (
        'script["src"] = "/local-runtime.js";',
        'script.setAttribute("src", "/local-runtime.js");',
    ),
)
def test_pages_tree_rejects_dynamic_source_bypasses(tmp_path: Path, bypass: str) -> None:
    site = copied_pages_tree(tmp_path)
    app = site / "app.js"
    app.write_text(app.read_text(encoding="utf-8") + "\n" + bypass, encoding="utf-8")

    assert "app.js: effective dynamic ORT source is not exact" in verify_pages_tree(site)
