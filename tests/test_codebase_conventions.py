"""Automated AST enforcement of OOP, naming, file-per-class, and one-way layer rules."""

from __future__ import annotations

import ast
from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "neurips_bci"

LAYER_RANK_BY_FAMILY: dict[str, int] = {
    "validator": 0,
    "operator": 1,
    "configuration": 2,
    "container": 3,
    "acquisition": 4,
    "adaptation": 5,
    "backbone": 6,
    "organ": 7,
    "decoder": 8,
    "evaluator": 9,
    "workflow": 10,
}

EXPECTED_SUFFIX_BY_FAMILY: dict[str, str] = {
    "validator": "Validator",
    "operator": "Operator",
    "configuration": "Configuration",
    "container": "Container",
    "acquisition": "Acquirer",
    "adaptation": "Adapter",
    "backbone": "Backbone",
    "organ": "Organ",
    "decoder": "Decoder",
    "evaluator": "Evaluator",
    "workflow": "Workflow",
}

FORBIDDEN_GENERIC_IDENTIFIERS = frozenset({"name", "data", "value", "result"})
CITATION_PATTERNS = (
    re.compile(r"\bet\s+al\.", re.IGNORECASE),
    re.compile(r"arxiv\.org", re.IGNORECASE),
    re.compile(r"doi\.org", re.IGNORECASE),
)


def _iter_class_module_files() -> list[Path]:
    return sorted(
        py_path
        for py_path in PACKAGE_ROOT.rglob("*.py")
        if py_path.name != "__init__.py"
    )


def test_one_class_per_file_and_family_suffix() -> None:
    violations: list[str] = []
    for py_path in _iter_class_module_files():
        rel_parts = py_path.relative_to(PACKAGE_ROOT).parts
        family_folder = rel_parts[0]
        expected_suffix = EXPECTED_SUFFIX_BY_FAMILY[family_folder]
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        top_classes = [stmt for stmt in tree.body if isinstance(stmt, ast.ClassDef)]
        if len(top_classes) != 1:
            violations.append(f"{py_path}: expected 1 top-level class, found {len(top_classes)}")
            continue
        cls_def = top_classes[0]
        if py_path.stem != cls_def.name:
            violations.append(f"{py_path}: filename {py_path.stem} != class {cls_def.name}")
        if not cls_def.name.endswith(expected_suffix):
            violations.append(f"{py_path}: class {cls_def.name} must end with {expected_suffix}")
        if len(rel_parts) == 2 and cls_def.name != expected_suffix:
            violations.append(
                f"{py_path}: family root class must be {expected_suffix}, got {cls_def.name}"
            )
        if len(rel_parts) == 3 and not rel_parts[1].endswith("s"):
            violations.append(f"{py_path}: sub-family folder {rel_parts[1]} must be plural")
    assert not violations, "\n".join(violations)


def test_one_way_layered_imports() -> None:
    violations: list[str] = []
    for py_path in PACKAGE_ROOT.rglob("*.py"):
        rel_parts = py_path.relative_to(PACKAGE_ROOT).parts
        if len(rel_parts) < 2:
            continue
        source_family = rel_parts[0]
        source_rank = LAYER_RANK_BY_FAMILY[source_family]
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for stmt in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(stmt, ast.Import):
                imported_modules.extend(alias_obj.name for alias_obj in stmt.names)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                imported_modules.append(stmt.module)
            for mod_str in imported_modules:
                if mod_str.startswith("neurips_bci."):
                    mod_parts = mod_str.split(".")
                    if len(mod_parts) >= 2 and mod_parts[1] in LAYER_RANK_BY_FAMILY:
                        target_family = mod_parts[1]
                        target_rank = LAYER_RANK_BY_FAMILY[target_family]
                        if target_rank > source_rank:
                            violations.append(
                                f"{py_path}: upward import {source_family} (L{source_rank}) -> {target_family} (L{target_rank})"
                            )
    assert not violations, "\n".join(violations)


def test_no_forbidden_generic_identifiers_or_paper_citations() -> None:
    violations: list[str] = []
    for py_path in _iter_class_module_files():
        source_text = py_path.read_text(encoding="utf-8")
        for regex_pat in CITATION_PATTERNS:
            if regex_pat.search(source_text):
                violations.append(f"{py_path}: contains paper citation matching {regex_pat.pattern}")
        tree = ast.parse(source_text, filename=str(py_path))
        for node_obj in ast.walk(tree):
            if isinstance(node_obj, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node_obj.name in FORBIDDEN_GENERIC_IDENTIFIERS:
                    violations.append(f"{py_path}:{node_obj.lineno}: forbidden definition '{node_obj.name}'")
            elif isinstance(node_obj, ast.arg):
                if node_obj.arg in FORBIDDEN_GENERIC_IDENTIFIERS:
                    violations.append(f"{py_path}:{node_obj.lineno}: forbidden argument '{node_obj.arg}'")
            elif isinstance(node_obj, ast.Name) and isinstance(node_obj.ctx, ast.Store):
                if node_obj.id in FORBIDDEN_GENERIC_IDENTIFIERS:
                    violations.append(f"{py_path}:{node_obj.lineno}: forbidden variable '{node_obj.id}'")
            elif isinstance(node_obj, ast.Attribute) and isinstance(node_obj.ctx, ast.Store):
                if node_obj.attr in FORBIDDEN_GENERIC_IDENTIFIERS or node_obj.attr.lstrip("_") in FORBIDDEN_GENERIC_IDENTIFIERS:
                    violations.append(f"{py_path}:{node_obj.lineno}: forbidden attribute '{node_obj.attr}'")
    assert not violations, "\n".join(violations)
