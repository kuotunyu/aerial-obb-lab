"""Fast, read-only checks for a fresh clone before publishing or pushing.

The checks themselves use the Python standard library; notebook source synchronization
invokes the Jupytext dev dependency installed by the lightweight default ``uv sync``.
Run this with the project's Python 3.11 interpreter.
"""

from __future__ import annotations

import ast
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import shutil
import subprocess
import sys
from threading import Thread
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

if __package__:
    from .pages_artifact_check import verify_pages_tree
else:
    from pages_artifact_check import verify_pages_tree

ROOT = Path(__file__).resolve().parents[1]

TOKEN_PATTERNS = {
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style token": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"(AKIA|ASIA)[A-Z0-9]{16}"),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    "PEM private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
TEXT_SUFFIXES = {
    ".css", ".html", ".js", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml",
}


def repository_files() -> list[Path]:
    if not (ROOT / ".git").exists():
        excluded = {".pytest_cache", ".venv", "__pycache__", "build", "dist"}
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(ROOT).parts)
        ]
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        text=False,
    )
    return [
        path
        for p in output.split(b"\0")
        if p and (path := ROOT / p.decode("utf-8")).is_file()
    ]


def check_python_syntax(files: list[Path]) -> None:
    python_files = [p for p in files if p.suffix == ".py"]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
    print(f"[OK] Python syntax: {len(python_files)} repository files")


def check_json(files: list[Path]) -> None:
    json_files = [p for p in files if p.suffix in {".json", ".ipynb"}]
    for path in json_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".ipynb":
            if data.get("nbformat") != 4:
                raise RuntimeError(f"unsupported notebook format: {path.relative_to(ROOT)}")
            for cell in data.get("cells", []):
                if cell.get("outputs"):
                    raise RuntimeError(f"notebook contains outputs: {path.relative_to(ROOT)}")
                if cell.get("execution_count") is not None:
                    raise RuntimeError(
                        f"notebook contains execution counts: {path.relative_to(ROOT)}"
                    )
    print(f"[OK] JSON/notebooks: {len(json_files)} repository files")


def check_notebook_pairs(files: list[Path]) -> None:
    notebook_files = {
        path.relative_to(ROOT)
        for path in files
        if path.parent == ROOT / "notebooks" and path.suffix in {".ipynb", ".py"}
    }
    missing: list[str] = []
    for path in sorted(notebook_files):
        partner = path.with_suffix(".py" if path.suffix == ".ipynb" else ".ipynb")
        if partner not in notebook_files:
            missing.append(f"{path} -> missing {partner}")
    if missing:
        raise RuntimeError("unpaired Jupytext notebooks:\n  " + "\n  ".join(missing))

    drifted: list[str] = []
    for notebook in sorted(path for path in notebook_files if path.suffix == ".ipynb"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupytext",
                "--to",
                "py:percent",
                "--output",
                "-",
                str(ROOT / notebook),
            ],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Jupytext sync check failed; run the preflight after `uv sync`:\n"
                + result.stderr.strip()
            )
        source_path = ROOT / notebook.with_suffix(".py")
        if result.stdout != source_path.read_text(encoding="utf-8"):
            drifted.append(f"{notebook} != {notebook.with_suffix('.py')}")
    if drifted:
        raise RuntimeError("Jupytext notebook/source drift:\n  " + "\n  ".join(drifted))
    print(f"[OK] Jupytext notebook pairs + source sync: {len(notebook_files) // 2}")


def markdown_prose(text: str) -> str:
    """Return Markdown prose while excluding fenced and inline code samples."""
    prose: list[str] = []
    fence_char: str | None = None
    for line in text.splitlines():
        fence = FENCE_RE.match(line)
        if fence:
            marker_char = fence.group(1)[0]
            if fence_char is None:
                fence_char = marker_char
            elif marker_char == fence_char:
                fence_char = None
            continue
        if fence_char is None:
            prose.append(INLINE_CODE_RE.sub("", line))
    return "\n".join(prose)


def check_markdown_links(files: list[Path]) -> None:
    checked = 0
    missing: list[str] = []
    for path in (p for p in files if p.suffix == ".md"):
        for raw in LINK_RE.findall(markdown_prose(path.read_text(encoding="utf-8"))):
            target = raw.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            checked += 1
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    if missing:
        raise RuntimeError("missing local Markdown links:\n  " + "\n  ".join(missing))
    print(f"[OK] Local Markdown links: {checked}")


def check_secrets(files: list[Path]) -> None:
    hits: list[str] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in TOKEN_PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(ROOT)}: {label}")
    if hits:
        raise RuntimeError("token-shaped strings found in tracked files:\n  " + "\n  ".join(hits))
    if not (ROOT / ".git").exists():
        if any(path.name == ".env" for path in files):
            raise RuntimeError(".env is present in the exported snapshot")
        print(f"[OK] Secret scan: {len(files)} clean-export files; Git history unavailable by design")
        return
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", ".env"], cwd=ROOT, check=False
    )
    if ignored.returncode != 0:
        raise RuntimeError(".env is not ignored by Git")

    combined = "|".join(f"({pattern.pattern})" for pattern in TOKEN_PATTERNS.values())
    commits = subprocess.check_output(
        ["git", "rev-list", "--all"], cwd=ROOT, text=True, encoding="utf-8"
    ).splitlines()
    historical_hits: set[str] = set()
    for commit in commits:
        result = subprocess.run(
            ["git", "grep", "-I", "-l", "-E", combined, commit, "--"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"git history secret scan failed at {commit[:8]}")
        historical_hits.update(line for line in result.stdout.splitlines() if line)
    if historical_hits:
        raise RuntimeError(
            "token-shaped strings found in Git history:\n  "
            + "\n  ".join(sorted(historical_hits))
        )
    print(f"[OK] Secret scan: working tree + {len(commits)} commits; .env is ignored")


def check_static_demo() -> None:
    folder = ROOT / "demo" / "web"
    required = {
        "index.html": 500,
        "app.js": 1_000,
        "obb.js": 1_000,
        "style.css": 500,
    }
    for name, minimum_size in required.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size < minimum_size:
            raise RuntimeError(f"static demo asset missing or too small: {path.relative_to(ROOT)}")
    html = (folder / "index.html").read_text(encoding="utf-8")
    for reference in ("style.css", "obb.js", "app.js"):
        if reference not in html:
            raise RuntimeError(f"index.html does not reference {reference}")
    for control in ('id="modelInput"', 'id="fileInput"', 'id="detectBtn"'):
        if control not in html:
            raise RuntimeError(f"index.html lacks BYOM control {control}")
    js = (folder / "app.js").read_text(encoding="utf-8")
    if re.search(r"\b(?:const|let|var)\s+MODEL_URL\b", js) or re.search(
        r'''["']models/[^"']+\.onnx["']''', js
    ):
        raise RuntimeError("app.js must not embed or fetch a model")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

    handler = partial(QuietHandler, directory=str(folder))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{base}/", timeout=5) as response:
            if response.status != 200 or b"Aerial OBB Lab" not in response.read():
                raise RuntimeError("HTTP validation failed for /")
        for name, minimum_size in required.items():
            with urlopen(f"{base}/{name}", timeout=5) as response:
                length = int(response.headers.get("Content-Length", "0"))
                if response.status != 200 or length < minimum_size:
                    raise RuntimeError(f"HTTP validation failed for {name}")
                if len(response.read()) < minimum_size:
                    raise RuntimeError(f"HTTP body validation failed for {name}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print("[OK] Static demo assets, references, and loopback HTTP serving")


def check_pages_artifact() -> None:
    errors = verify_pages_tree(ROOT / "demo" / "web")
    if errors:
        raise RuntimeError("Pages artifact boundary:\n  " + "\n  ".join(errors))
    print("[OK] Pages artifact boundary")


def check_javascript() -> None:
    node = shutil.which("node")
    if not node:
        print("[SKIP] JavaScript syntax: Node.js not installed")
        return
    for relative in ("demo/web/obb.js", "demo/web/app.js"):
        subprocess.run([node, "--check", str(ROOT / relative)], cwd=ROOT, check=True)
    subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "browser_parity_runner.js"),
            str(ROOT / "tests" / "fixtures" / "browser_parity.json"),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print("[OK] JavaScript syntax + browser parity")


def main() -> int:
    try:
        files = repository_files()
        check_python_syntax(files)
        check_notebook_pairs(files)
        check_json(files)
        check_markdown_links(files)
        check_secrets(files)
        check_pages_artifact()
        check_static_demo()
        check_javascript()
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("Repository preflight: ALL GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
