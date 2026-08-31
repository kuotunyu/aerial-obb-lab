"""Verify the final real-demo-only ``demo/web`` privacy boundary.

This checker is deliberately read-only and Python-standard-library-only so it can
run in repository preflight and admit only the reviewed public asset inventory.
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
)
ALLOWED_FILES = frozenset(REQUIRED_FILES)
ALLOWED_DIRECTORIES = frozenset(("fixtures", "fonts", "models", "samples", "third_party"))
FONT_PATH = "fonts/IBMPlexSansCondensed-SemiBold.woff2"
MODEL_PATH = "models/yolo26n-obb-privacy-sanitized.onnx"
IMAGE_PATH = "samples/boats.jpg"
BINARY_PUBLIC_PATHS = frozenset((FONT_PATH, MODEL_PATH, IMAGE_PATH))
SOURCE_MODEL_SHA256 = "02f7c539600296d7389341280beb82da810b15dc09c54cf2bc70f7f610331b38"
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".svg", ".txt"}
RUNTIME_TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".svg"}
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
ABSOLUTE_USER_PATH_BYTES_RE = re.compile(
    rb"(?i)(?:[a-z]:[\\/]+(?:users|documents[ ]and[ ]settings)[\\/]+|/(?:users|home)/)"
)
URL_RE = re.compile(r'''(?:(?:https?:)?//)[^\s"'`<>()\[\]{}]+''', re.I)
GITHUB_NAVIGATION_RE = re.compile(
    r'''<a\b[^>]*\bhref\s*=\s*["'](https://github\.com/[^"']+)["'][^>]*>''',
    re.I,
)
SVG_NAMESPACE_RE = re.compile(
    r'''\bxmlns\s*=\s*["'](http://www\.w3\.org/2000/svg)["']''', re.I
)
FORBIDDEN_BROWSER_API_RE = re.compile(
    r"\b(?:localStorage|sessionStorage|indexedDB|caches|XMLHttpRequest|"
    r"WebSocket|EventSource|sendBeacon|serviceWorker)\b|\bnavigator\.storage\b"
)
REMOTE_MODEL_FETCH_RE = re.compile(
    r'''\bfetch\s*\(\s*["'`][^"'`]*(?:\.ckpt|\.engine|\.mlpackage|\.onnx|\.pb|'''
    r'''\.pt|\.pth|\.safetensors|\.tflite|\.torchscript|\.weights)(?:[?#][^"'`]*)?["'`]''',
    re.I,
)

ORT_PACKAGE_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/"
ORT_SCRIPT_URL = ORT_PACKAGE_BASE + "ort.min.js"
ORT_INTEGRITY = (
    "sha384-RPL/K8tc0JVaNWsunkEmCzLeieefvFX2UCRLKLmLVChCI6P+CTKhzqF7VIeCc3Zp"
)
REVIEWED_ASSET_DIGESTS = {
    "fonts/IBMPlexSansCondensed-SemiBold.woff2": (
        "385a082a1eac88343eab01fb6746be04b7175dacaf4550b17dee76ea0f78126d",
        "reviewed font bytes differ",
    ),
    IMAGE_PATH: (
        "8c5ada657cf8110a9f8aaac954c1dd96cde0187315b581276c32b0d1863e756f",
        "reviewed sample image bytes differ",
    ),
    MODEL_PATH: (
        "a0a1a2dd357067e8c6c9f5ce7bb33487188423f9722e813be880da4f9badcd97",
        "published model bytes differ",
    ),
}
REVIEWED_TEXT_DIGESTS = {
    "index.html": (
        "616fc0252410212fe5406953308e50cfc597cd92ebf0ade27b1785e86066a5a9",
        "reviewed HTML bytes differ",
    ),
    "app.js": (
        "3b4de4978d411bda06c90392b99258993ed69c031737ecaecaa98099ef7bf7e6",
        "reviewed application bytes differ",
    ),
    "obb.js": (
        "c2c83882a1cb1b6d76ab48d12784af6c2ef526be08d4fe1b7ed23798b7350043",
        "reviewed geometry bytes differ",
    ),
    "demo-assets.js": (
        "8d66746927b37b0e22c2c292c01e3972e17e9302cbeee3c8b8c86ed9540f3767",
        "reviewed demo asset loader bytes differ",
    ),
    "style.css": (
        "70cfd7920c508d3e833a865b467653086a9a5e443f8d98a2a1bd0b01bbda961c",
        "reviewed stylesheet bytes differ",
    ),
    "fonts/IBM-Plex-OFL.txt": (
        "9590325331b1975eac408dc78e7d369c042f565cee8aa9e34d6b40524f400972",
        "reviewed font license bytes differ",
    ),
    "README.md": (
        "8bc13d6e4c6cd3528be42a874e1809f23d0127d5f757811d295a58df947308a7",
        "reviewed README bytes differ",
    ),
    "demo-model.json": (
        "9dab4fa0b93ae4f4fabc1467af032535485f84dd2df11aefa5f27d1ab38d5f54",
        "reviewed demo manifest bytes differ",
    ),
    "third_party/ULTRALYTICS-AGPL-3.0.txt": (
        "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
        "reviewed Ultralytics license bytes differ",
    ),
    "third_party/yolo26n-obb-privacy-sanitization.json": (
        "e0e03b45e5750ebe21070e93ef4f2c537f6040f471980f55428aa4d69d7a659b",
        "reviewed sanitization record bytes differ",
    ),
    "THIRD_PARTY_NOTICES.md": (
        "3988f1b9b5bb47a95067210af64dd4088ffb161df747d10ff9e99779ae69de07",
        "reviewed third-party notice bytes differ",
    ),
}

APPROVED_PROVENANCE_URLS = frozenset(
    (
        "https://ultralytics.com/images/boats.jpg",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.onnx",
    )
)


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, OSError):
        return False
    reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _canonical_lf_text_sha256(path: Path) -> str:
    """Hash UTF-8 text content after normalizing CRLF and lone CR to LF."""
    text = path.read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_public_text(path: Path, relative: str, errors: list[str]) -> str | None:
    try:
        payload = path.read_bytes()
    except OSError:
        errors.append(f"{relative}: cannot read file")
        return None
    if b"\0" in payload:
        errors.append(f"{relative}: unexpected binary file")
        return None
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{relative}: unexpected binary file")
        return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _check_model_binary(binary: bytes, digest: str, errors: list[str]) -> None:
    if ABSOLUTE_USER_PATH_BYTES_RE.search(binary):
        errors.append(f"{MODEL_PATH}: binary contains absolute user path")
    if digest == SOURCE_MODEL_SHA256:
        errors.append(f"{MODEL_PATH}: upstream model bytes are forbidden")


def _scan_runtime_urls(relative: str, text: str, errors: list[str]) -> None:
    approved_context_spans = (
        {match.span(1) for match in GITHUB_NAVIGATION_RE.finditer(text)}
        if relative.endswith(".html")
        else set()
    )
    if relative.endswith(".svg"):
        approved_context_spans.update(
            match.span(1) for match in SVG_NAMESPACE_RE.finditer(text)
        )
    for match in URL_RE.finditer(text):
        url = match.group(0)
        if url.startswith(ORT_PACKAGE_BASE):
            continue
        if relative in {
            "demo-model.json",
            "third_party/yolo26n-obb-privacy-sanitization.json",
        } and url in APPROVED_PROVENANCE_URLS:
            continue
        if match.span() in approved_context_spans:
            continue
        errors.append(f"{relative}: unapproved external origin: {url}")


def _strip_javascript_comments(text: str) -> str:
    """Blank JavaScript comments while preserving strings and line positions."""
    output: list[str] = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if quote is not None:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'", "`"}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            output.extend((" ", " "))
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                output.append(" ")
                index += 1
            continue
        if char == "/" and following == "*":
            output.extend((" ", " "))
            index += 2
            while index < len(text):
                if index + 1 < len(text) and text[index : index + 2] == "*/":
                    output.extend((" ", " "))
                    index += 2
                    break
                output.append(text[index] if text[index] in "\r\n" else " ")
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _active_line_count(source: str, line: str) -> int:
    return len(re.findall(rf"^[ \t]*{re.escape(line)}[ \t]*$", source, re.M))


SCRIPT_SOURCE_SETTER_RE = re.compile(
    r'''\bscript\s*(?:\.\s*src\s*=|\[\s*["']src["']\s*\]\s*=|'''
    r'''\.\s*setAttribute\s*\(\s*["']src["'])'''
)
SCRIPT_INTEGRITY_SETTER_RE = re.compile(
    r'''\bscript\s*(?:\.\s*integrity\s*=|\[\s*["']integrity["']\s*\]\s*=|'''
    r'''\.\s*setAttribute\s*\(\s*["']integrity["'])'''
)
SCRIPT_CORS_SETTER_RE = re.compile(
    r'''\bscript\s*(?:\.\s*crossOrigin\s*=|\[\s*["']crossOrigin["']\s*\]\s*=|'''
    r'''\.\s*setAttribute\s*\(\s*["']crossorigin["'])''',
    re.I,
)
WASM_PATH_SETTER_RE = re.compile(
    r"\b(?:globalThis\.)?ort\.env\.wasm\.wasmPaths\s*="
)


def _check_exact_runtime_contract(root: Path, texts: dict[str, str], errors: list[str]) -> None:
    app = texts.get("app.js")
    if app is not None:
        active_app = _strip_javascript_comments(app)
        required_constant_lines = (
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
        )
        for line, reason in required_constant_lines:
            if _active_line_count(active_app, line) != 1:
                errors.append(f"app.js: {reason}")

        source_line = "script.src = ORT_URL;"
        integrity_line = "script.integrity = ORT_INTEGRITY;"
        cors_line = 'script.crossOrigin = "anonymous";'
        wasm_line = "globalThis.ort.env.wasm.wasmPaths = ORT_WASM_BASE;"
        source_line_count = _active_line_count(active_app, source_line)
        integrity_line_count = _active_line_count(active_app, integrity_line)
        cors_line_count = _active_line_count(active_app, cors_line)
        wasm_line_count = _active_line_count(active_app, wasm_line)
        source_setters = len(SCRIPT_SOURCE_SETTER_RE.findall(active_app))
        integrity_setters = len(SCRIPT_INTEGRITY_SETTER_RE.findall(active_app))
        cors_setters = len(SCRIPT_CORS_SETTER_RE.findall(active_app))
        wasm_setters = len(WASM_PATH_SETTER_RE.findall(active_app))

        if source_line_count != 1:
            errors.append("app.js: dynamic ORT script assignment is missing")
        if source_line_count != 1 or source_setters != 1:
            errors.append("app.js: effective dynamic ORT source is not exact")
        if integrity_line_count != 1:
            errors.append("app.js: dynamic ORT integrity assignment is missing")
        if cors_line_count != 1:
            errors.append("app.js: exact ORT anonymous CORS setting is missing")
        if wasm_line_count != 1:
            errors.append("app.js: exact ORT WASM path assignment is missing")

        constant_assignment_counts = (
            len(re.findall(r"\bORT_URL\s*=", active_app)),
            len(re.findall(r"\bORT_WASM_BASE\s*=", active_app)),
            len(re.findall(r"\bORT_INTEGRITY\s*=", active_app)),
        )
        if any(
            count > 1
            for count in (
                *constant_assignment_counts,
                source_setters,
                integrity_setters,
                cors_setters,
                wasm_setters,
            )
        ):
            errors.append("app.js: effective ORT tuple is overridden")

    html = texts.get("index.html")
    if html is not None:
        for reference in ("style.css", "obb.js", "demo-assets.js", "app.js"):
            if reference not in html:
                errors.append(f"index.html: required reference is missing: {reference}")

    demo_assets = texts.get("demo-assets.js")
    model_reference = 'path: "models/yolo26n-obb-privacy-sanitized.onnx"'
    if demo_assets is not None and model_reference not in demo_assets:
        errors.append("demo-assets.js: exact derivative model path is missing")

    style = texts.get("style.css")
    font_reference = 'url("fonts/IBMPlexSansCondensed-SemiBold.woff2")'
    if style is not None and font_reference not in style:
        errors.append("style.css: exact reviewed font reference is missing")

    license_text = texts.get("fonts/IBM-Plex-OFL.txt")
    if license_text is not None and "SIL OPEN FONT LICENSE Version 1.1" not in license_text:
        errors.append("fonts/IBM-Plex-OFL.txt: SIL Open Font License 1.1 text is missing")

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
            if relative not in ALLOWED_DIRECTORIES:
                errors.append(f"{relative}: unexpected Pages directory")
            continue
        if not path.is_file():
            errors.append(f"{relative}: unsupported filesystem entry")
            continue

        try:
            stat = path.stat()
        except OSError:
            errors.append(f"{relative}: cannot inspect file")
            continue
        if stat.st_nlink != 1:
            errors.append(f"{relative}: hard link count is {stat.st_nlink}")
        if stat.st_size > ONE_MIB and relative not in {FONT_PATH, MODEL_PATH}:
            errors.append(f"{relative}: file exceeds 1 MiB")

        suffix = path.suffix.casefold()
        if (
            (suffix in FORBIDDEN_MODEL_SUFFIXES and relative != MODEL_PATH)
            or suffix in FORBIDDEN_ARCHIVE_SUFFIXES
        ):
            errors.append(f"{relative}: forbidden model/runtime artifact")
        if FORBIDDEN_PATH_TOKEN_RE.search(relative):
            errors.append(f"{relative}: forbidden DOTA-derived path")
        if relative not in ALLOWED_FILES:
            errors.append(f"{relative}: unexpected Pages file")

        reviewed = REVIEWED_ASSET_DIGESTS.get(relative)
        if reviewed is not None:
            expected_digest, reason = reviewed
            try:
                actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                errors.append(f"{relative}: cannot read file")
            else:
                if actual_digest != expected_digest:
                    errors.append(f"{relative}: {reason}")

        reviewed_text = REVIEWED_TEXT_DIGESTS.get(relative)
        if reviewed_text is not None:
            expected_digest, reason = reviewed_text
            try:
                actual_digest = _canonical_lf_text_sha256(path)
            except OSError:
                errors.append(f"{relative}: cannot read file")
            except UnicodeDecodeError:
                errors.append(f"{relative}: {reason}")
            else:
                if actual_digest != expected_digest:
                    errors.append(f"{relative}: {reason}")

        if relative in BINARY_PUBLIC_PATHS:
            if relative == MODEL_PATH:
                try:
                    binary = path.read_bytes()
                except OSError:
                    errors.append(f"{relative}: cannot read file")
                else:
                    _check_model_binary(binary, hashlib.sha256(binary).hexdigest(), errors)
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
        if "synthetic" in text.casefold():
            errors.append(f"{relative}: current Synthetic reference is forbidden")
        if suffix in RUNTIME_TEXT_SUFFIXES:
            _scan_runtime_urls(relative, text, errors)
            if REMOTE_MODEL_FETCH_RE.search(text):
                errors.append(f"{relative}: unapproved runtime/model reference")
            if FORBIDDEN_BROWSER_API_RE.search(text):
                errors.append(f"{relative}: forbidden browser storage/network API")

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
