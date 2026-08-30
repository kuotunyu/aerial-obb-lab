"""Verify the exact, publishable ``demo/web`` privacy boundary.

This checker is deliberately read-only and Python-standard-library-only so it can
run in repository preflight and directly against a staged Pages tree.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import stat as stat_module
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES_ROOT = ROOT / "demo" / "web"

ONE_MIB = 1024 * 1024
REQUIRED_FILES = (
    "index.html",
    "app.js",
    "obb.js",
    "showcase-fixture.js",
    "style.css",
    "fixtures/showcase.svg",
    "fonts/IBMPlexSansCondensed-SemiBold.woff2",
    "fonts/IBM-Plex-OFL.txt",
)
FONT_PATH = "fonts/IBMPlexSansCondensed-SemiBold.woff2"
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".svg", ".txt"}
RUNTIME_TEXT_SUFFIXES = {".css", ".html", ".js"}
FORBIDDEN_MODEL_SUFFIXES = {
    ".ckpt",
    ".engine",
    ".mlpackage",
    ".onnx",
    ".pb",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".torchscript",
    ".weights",
}
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}
FORBIDDEN_PATH_TOKEN_RE = re.compile(r"(?:dota|hbb_vs_obb)", re.I)
TOKEN_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
ABSOLUTE_USER_PATH_RE = re.compile(
    r'''(?ix)
    (?:
        [A-Z]:[\\/]+(?:Users|Documents[ ]and[ ]Settings)[\\/]+[^\\/\s"']+
        |
        /(?:Users|home)/[^/\s"']+
    )
    '''
)
URL_RE = re.compile(r'''https?://[^\s"'`<>()\[\]{}]+''', re.I)
GITHUB_NAVIGATION_RE = re.compile(
    r'''<a\b[^>]*\bhref\s*=\s*["'](https://github\.com/[^"']+)["'][^>]*>''',
    re.I,
)

ORT_PACKAGE_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_SCRIPT_URL = ORT_PACKAGE_BASE + "ort.min.js"
ORT_INTEGRITY = (
    "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
)
REVIEWED_ASSET_DIGESTS = {
    "fixtures/showcase.svg": (
        "c208b1a056555825d75f25a421403a11738fb2efa90a880845a79e3af5c35385",
        "reviewed synthetic fixture bytes differ",
    ),
    "fonts/IBMPlexSansCondensed-SemiBold.woff2": (
        "385a082a1eac88343eab01fb6746be04b7175dacaf4550b17dee76ea0f78126d",
        "reviewed font bytes differ",
    ),
    "fonts/IBM-Plex-OFL.txt": (
        "aaa43b32d5a6ea1aa1a8768ecb85899b22f94c05486c743838610f5e640abebc",
        "reviewed font license bytes differ",
    ),
}


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _read_public_text(path: Path, relative: str, errors: list[str]) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        errors.append(f"{relative}: cannot read file ({exc})")
        return None
    if b"\0" in payload:
        errors.append(f"{relative}: unexpected binary file")
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{relative}: unexpected binary file")
        return None


def _scan_runtime_urls(relative: str, text: str, errors: list[str]) -> None:
    reviewed_navigation = (
        set(GITHUB_NAVIGATION_RE.findall(text)) if relative.endswith(".html") else set()
    )
    for url in URL_RE.findall(text):
        if url.startswith(ORT_PACKAGE_BASE):
            continue
        if url in reviewed_navigation:
            continue
        errors.append(f"{relative}: unapproved external origin: {url}")


def _check_exact_runtime_contract(root: Path, texts: dict[str, str], errors: list[str]) -> None:
    app = texts.get("app.js")
    if app is not None:
        required_app_fragments = (
            (
                f'const ORT_URL = "{ORT_SCRIPT_URL}";',
                "exact ORT script URL is missing",
            ),
            (
                f'const ORT_WASM_BASE = "{ORT_PACKAGE_BASE}";',
                "exact ORT WASM base URL is missing",
            ),
            (
                f'const ORT_INTEGRITY = "{ORT_INTEGRITY}";',
                "exact ORT integrity is missing",
            ),
            ("script.src = ORT_URL;", "dynamic ORT script assignment is missing"),
            ("script.integrity = ORT_INTEGRITY;", "dynamic ORT integrity assignment is missing"),
            (
                'script.crossOrigin = "anonymous";',
                "exact ORT anonymous CORS setting is missing",
            ),
            (
                "globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;",
                "exact ORT WASM path assignment is missing",
            ),
        )
        for fragment, reason in required_app_fragments:
            if fragment not in app:
                errors.append(f"app.js: {reason}")

    html = texts.get("index.html")
    if html is not None:
        for reference in ("style.css", "obb.js", "showcase-fixture.js", "app.js"):
            if reference not in html:
                errors.append(f"index.html: required reference is missing: {reference}")

    fixture = texts.get("showcase-fixture.js")
    if fixture is not None and 'imageUrl: "fixtures/showcase.svg"' not in fixture:
        errors.append("showcase-fixture.js: exact synthetic fixture reference is missing")

    style = texts.get("style.css")
    font_reference = 'url("fonts/IBMPlexSansCondensed-SemiBold.woff2")'
    if style is not None and font_reference not in style:
        errors.append("style.css: exact reviewed font reference is missing")

    license_text = texts.get("fonts/IBM-Plex-OFL.txt")
    if license_text is not None and "SIL OPEN FONT LICENSE Version 1.1" not in license_text:
        errors.append("fonts/IBM-Plex-OFL.txt: SIL Open Font License 1.1 text is missing")

    fixture_path = root / "fixtures" / "showcase.svg"
    if fixture_path.is_file() and fixture_path.is_symlink():
        errors.append("fixtures/showcase.svg: required fixture must be a regular file")


def verify_pages_tree(root: Path) -> list[str]:
    """Return deterministic ``path: reason`` violations without modifying *root*."""
    root = Path(root)
    errors: list[str] = []
    texts: dict[str, str] = {}

    if _is_link(root):
        return [".: symbolic link"]
    if not root.is_dir():
        return [".: Pages root is missing or is not a directory"]

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or _is_link(path):
            errors.append(f"{relative}: required Pages file is missing")

    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if _is_link(path):
            errors.append(f"{relative}: symbolic link")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            errors.append(f"{relative}: unsupported filesystem entry")
            continue

        try:
            stat = path.stat()
        except OSError as exc:
            errors.append(f"{relative}: cannot inspect file ({exc})")
            continue
        if stat.st_nlink != 1:
            errors.append(f"{relative}: hard link count is {stat.st_nlink}")
        if stat.st_size > ONE_MIB and relative != FONT_PATH:
            errors.append(f"{relative}: file exceeds 1 MiB")

        suffix = path.suffix.casefold()
        if suffix in FORBIDDEN_MODEL_SUFFIXES or suffix in FORBIDDEN_ARCHIVE_SUFFIXES:
            errors.append(f"{relative}: forbidden model/runtime artifact")
        if FORBIDDEN_PATH_TOKEN_RE.search(relative):
            errors.append(f"{relative}: forbidden DOTA-derived path")

        reviewed = REVIEWED_ASSET_DIGESTS.get(relative)
        if reviewed is not None:
            expected_digest, reason = reviewed
            try:
                actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                errors.append(f"{relative}: cannot read file ({exc})")
            else:
                if actual_digest != expected_digest:
                    errors.append(f"{relative}: {reason}")

        if relative == FONT_PATH:
            continue
        if suffix not in TEXT_SUFFIXES:
            errors.append(f"{relative}: unexpected binary file")
            continue

        text = _read_public_text(path, relative, errors)
        if text is None:
            continue
        texts[relative] = text
        if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
            errors.append(f"{relative}: token-shaped string")
        if ABSOLUTE_USER_PATH_RE.search(text):
            errors.append(f"{relative}: absolute user path")
        if suffix in RUNTIME_TEXT_SUFFIXES:
            _scan_runtime_urls(relative, text, errors)

    _check_exact_runtime_contract(root, texts, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_PAGES_ROOT,
        help="Pages tree to inspect (default: demo/web)",
    )
    args = parser.parse_args(argv)
    errors = verify_pages_tree(args.root)
    if errors:
        print("[FAIL] Pages artifact boundary:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("[OK] Pages artifact boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
