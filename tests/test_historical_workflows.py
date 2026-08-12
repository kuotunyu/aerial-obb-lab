from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCES = (
    ROOT / "notebooks" / "01_train_dotav1_a100.py",
    ROOT / "notebooks" / "02_benchmark_colab.py",
    ROOT / "notebooks" / "03_recover_per_class_metrics_colab.py",
)
REMOTE_NOTEBOOK_SOURCES = NOTEBOOK_SOURCES[:2]
RISKY_IMPORT_ROOTS = {"google", "huggingface_hub", "torch", "ultralytics"}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _false_assignment_line(tree: ast.Module, name: str) -> int:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            assert isinstance(node.value, ast.Constant) and node.value.value is False
            return node.lineno
    raise AssertionError(f"missing default-false {name}")


def _fail_closed_guard_line(tree: ast.Module, name: str) -> int:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        is_not_name = (
            isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and node.test.operand.id == name
        )
        raises = any(isinstance(child, ast.Raise) for child in node.body)
        if is_not_name and raises:
            return node.lineno
    raise AssertionError(f"missing fail-closed guard for {name}")


def _first_risky_import_line(tree: ast.Module) -> int:
    lines: list[int] = []
    for node in ast.walk(tree):
        roots: set[str] = set()
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = {node.module.split(".", 1)[0]}
        if roots & RISKY_IMPORT_ROOTS:
            lines.append(node.lineno)
    assert lines, "fixture must contain a historical heavyweight import"
    return min(lines)


def _import_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_notebook_sources_require_explicit_historical_gpu_ack_before_risky_imports() -> None:
    for path in NOTEBOOK_SOURCES:
        tree = _parse(path)
        assignment = _false_assignment_line(tree, "ALLOW_HISTORICAL_GPU_RUN")
        guard = _fail_closed_guard_line(tree, "ALLOW_HISTORICAL_GPU_RUN")
        first_risky_import = _first_risky_import_line(tree)

        assert assignment < guard < first_risky_import, path.name


def test_remote_notebook_sources_require_separate_write_ack_before_risky_imports() -> None:
    for path in REMOTE_NOTEBOOK_SOURCES:
        tree = _parse(path)
        assignment = _false_assignment_line(tree, "ALLOW_REMOTE_WRITES")
        guard = _fail_closed_guard_line(tree, "ALLOW_REMOTE_WRITES")
        first_risky_import = _first_risky_import_line(tree)

        assert assignment < guard < first_risky_import, path.name


def test_historical_smoke_refuses_to_start_without_acknowledgement(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "smoke_test.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    proc = subprocess.run(
        [sys.executable, "-S", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 2
    assert "--acknowledge-historical-gpu-workflow" in output


def test_historical_smoke_has_no_hugging_face_write_client() -> None:
    tree = _parse(ROOT / "scripts" / "smoke_test.py")

    assert "huggingface_hub" not in _import_roots(tree)
