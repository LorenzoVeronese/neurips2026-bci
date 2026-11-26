"""Static AST and SHA-256 archive safety validator for competition bundles."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import zipfile

from neurips_bci.validator.type_validators.ParameterTypeValidator import ParameterTypeValidator
from neurips_bci.validator.type_validators.TypeValidator import TypeValidator


class ArchiveSafetyTypeValidator(TypeValidator):
    """Verifies flat layout, SHA-256 manifest integrity, zero pickle loads, and zero network imports."""

    _FORBIDDEN_IMPORTS = frozenset({"urllib", "requests", "httpx", "aiohttp", "ftplib", "telnetlib", "pickle"})

    def __init__(self, target_archive_path: str | Path) -> None:
        super().__init__(validator_scope="submission_archive_safety")
        self._target_archive_path = ParameterTypeValidator.require_path(
            target_archive_path, "target_archive_path", must_exist=True
        )

    @property
    def target_archive_path(self) -> Path:
        return self._target_archive_path

    def verify_contract(self) -> bool:
        inspection_report = self.inspect_archive()
        if not inspection_report["all_checks_passed"]:
            raise ValueError(f"Archive failed safety checks: {inspection_report['violations']}")
        return True

    def inspect_archive(self) -> dict[str, object]:
        violations: list[str] = []
        if self._target_archive_path.is_dir():
            entry_paths = sorted(p for p in self._target_archive_path.rglob("*") if p.is_file())
            relative_entries = [str(p.relative_to(self._target_archive_path)) for p in entry_paths]
            byte_reader = lambda rel_key: (self._target_archive_path / rel_key).read_bytes()
        else:
            with zipfile.ZipFile(self._target_archive_path, "r") as zip_handle:
                relative_entries = sorted(zip_handle.namelist())
                cached_bytes = {entry_key: zip_handle.read(entry_key) for entry_key in relative_entries}
            byte_reader = lambda rel_key: cached_bytes[rel_key]

        for entry_key in relative_entries:
            if "/" in entry_key or "\\" in entry_key:
                violations.append(f"Nested path not permitted in flat bundle: {entry_key}")
            if entry_key.endswith((".pkl", ".pickle", ".joblib")):
                violations.append(f"Forbidden pickle file in bundle: {entry_key}")

        python_entries = [entry_key for entry_key in relative_entries if entry_key.endswith(".py")]
        if python_entries != ["submission.py"]:
            violations.append(f"Expected exactly ['submission.py'], found {python_entries}")
        else:
            submission_source = byte_reader("submission.py").decode("utf-8")
            violations.extend(self.audit_python_source(submission_source))

        if "MANIFEST_SHA256.json" in relative_entries:
            manifest_payload = json.loads(byte_reader("MANIFEST_SHA256.json").decode("utf-8"))
            recorded_hashes = manifest_payload.get("files", manifest_payload)
            if isinstance(recorded_hashes, dict):
                for file_key, file_meta in recorded_hashes.items():
                    if file_key == "MANIFEST_SHA256.json" or file_key not in relative_entries:
                        continue
                    expected_digest = (
                        file_meta["sha256"] if isinstance(file_meta, dict) and "sha256" in file_meta else file_meta
                    )
                    if isinstance(expected_digest, str):
                        actual_digest = hashlib.sha256(byte_reader(file_key)).hexdigest()
                        if actual_digest != expected_digest:
                            violations.append(f"SHA-256 mismatch for {file_key}")

        return {
            "archive_path": str(self._target_archive_path),
            "entry_count": len(relative_entries),
            "violations": violations,
            "all_checks_passed": len(violations) == 0,
        }

    @classmethod
    def audit_python_source(cls, python_source_text: str) -> list[str]:
        syntax_tree = ast.parse(python_source_text)
        source_violations: list[str] = []
        for syntax_node in ast.walk(syntax_tree):
            if isinstance(syntax_node, ast.Import):
                for alias_item in syntax_node.names:
                    top_module = alias_item.name.split(".")[0]
                    if top_module in cls._FORBIDDEN_IMPORTS:
                        source_violations.append(f"Forbidden import: {top_module}")
            elif isinstance(syntax_node, ast.ImportFrom) and syntax_node.module:
                top_module = syntax_node.module.split(".")[0]
                if top_module in cls._FORBIDDEN_IMPORTS:
                    source_violations.append(f"Forbidden from-import: {top_module}")
            elif isinstance(syntax_node, ast.keyword):
                if syntax_node.arg == "allow_pickle" and isinstance(syntax_node.value, ast.Constant):
                    if syntax_node.value.value is True:
                        source_violations.append("allow_pickle=True is forbidden")
                if syntax_node.arg == "weights_only" and isinstance(syntax_node.value, ast.Constant):
                    if syntax_node.value.value is False:
                        source_violations.append("weights_only=False is forbidden")
        return source_violations
