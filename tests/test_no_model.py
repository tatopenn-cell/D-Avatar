"""Test: the default path must not import any ML library or model file."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

DA_AVATAR = Path(__file__).resolve().parents[1] / 'd_avatar'

_FORBIDDEN_IMPORTS = (
    'mediapipe',
    'torch',
    'tensorflow',
    'dlib',
    'onnxruntime',
    'cv2.dnn',
)

_FORBIDDEN_EXTENSIONS = ('.task', '.onnx', '.tflite', '.pt')

_MODULES = (
    'd_avatar.tracker',
    'd_avatar.avatar',
    'd_avatar.vmc',
    'd_avatar.pipeline',
    'd_avatar.capture',
    'd_avatar.params',
)


def _source_tree() -> Path:
    return Path(__file__).resolve().parents[1]


def _imports_forbidden(module_name: str) -> list[str]:
    """Return forbidden imports found in a module's source (without importing it)."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return []
    mod_file = mod.__file__
    if mod_file is None:
        return []
    file = Path(mod_file)
    tree = ast.parse(file.read_text(encoding='utf-8'))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or '').startswith(_FORBIDDEN_IMPORTS) or any(
                (node.module or '').startswith(f) for f in _FORBIDDEN_IMPORTS
            ):
                found.append(node.module or '')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(f) for f in _FORBIDDEN_IMPORTS):
                    found.append(alias.name)
    return found


def _model_files() -> list[Path]:
    """Only the default-path package (d_avatar/) may not contain model files."""
    hits: list[Path] = []
    for ext in _FORBIDDEN_EXTENSIONS:
        hits.extend(DA_AVATAR.rglob(f'*{ext}'))
    return hits


def test_no_forbidden_imports_in_default_path() -> None:
    """No module in the default path may import an ML library."""
    for mod in _MODULES:
        hits = _imports_forbidden(mod)
        assert not hits, f'{mod} imports forbidden ML library: {hits}'


def test_no_model_files_in_repo() -> None:
    """No .task / .onnx / .tflite / .pt file may be in the repository."""
    hits = _model_files()
    assert not hits, f'Forbidden model files found: {[str(p) for p in hits]}'


def test_tracker_fails_if_mediapipe_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a mediapipe import to tracker.py must make the import test fail."""
    tracker = DA_AVATAR / 'tracker.py'
    original = tracker.read_text(encoding='utf-8')
    try:
        tracker.write_text('import mediapipe\n' + original, encoding='utf-8')
        hits = _imports_forbidden('d_avatar.tracker')
        assert 'mediapipe' in hits, 'expected mediapipe import to be detected'
    finally:
        tracker.write_text(original, encoding='utf-8')
