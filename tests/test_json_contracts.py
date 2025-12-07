from __future__ import annotations

# ruff: noqa: S101 - assertions express expectations in test cases
import copy
import json
from pathlib import Path
from typing import cast

from x_make_common_x.json_contracts import validate_payload, validate_schema

from x_make_pip_updates_x.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)
from x_make_pip_updates_x.update_flow import main_json

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "json_contracts"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _load_fixture(name: str) -> dict[str, object]:
    path = FIXTURE_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    if not isinstance(loaded, dict):
        message = f"Fixture payload must be an object: {name}"
        raise TypeError(message)
    payload = cast("dict[str, object]", loaded)
    return {str(key): value for key, value in payload.items()}


SAMPLE_INPUT = _load_fixture("input")
SAMPLE_OUTPUT = _load_fixture("output")
SAMPLE_ERROR = _load_fixture("error")


def test_schemas_are_valid() -> None:
    for schema in (INPUT_SCHEMA, OUTPUT_SCHEMA, ERROR_SCHEMA):
        validate_schema(schema)


def test_sample_payloads_match_schema() -> None:
    validate_payload(SAMPLE_INPUT, INPUT_SCHEMA)
    validate_payload(SAMPLE_OUTPUT, OUTPUT_SCHEMA)
    validate_payload(SAMPLE_ERROR, ERROR_SCHEMA)


def test_existing_reports_align_with_schema() -> None:
    report_files = sorted(REPORTS_DIR.glob("x_make_pip_updates_x_run_*.json"))
    assert report_files, "expected at least one pip-updates run report to validate"
    for report_file in report_files:
        with report_file.open("r", encoding="utf-8") as handle:
            loaded: object = json.load(handle)
        if not isinstance(loaded, dict):
            message = f"Report payload must be an object: {report_file}"
            raise TypeError(message)
        typed_payload = cast("dict[str, object]", loaded)
        validate_payload(typed_payload, OUTPUT_SCHEMA)


def test_main_json_executes_happy_path() -> None:
    payload = copy.deepcopy(SAMPLE_INPUT)
    result = main_json(payload)
    validate_payload(result, OUTPUT_SCHEMA)
    assert result["status"] in {"success", "error"}


def test_main_json_returns_error_for_invalid_payload() -> None:
    invalid = copy.deepcopy(SAMPLE_INPUT)
    parameters = invalid.get("parameters")
    if isinstance(parameters, dict):
        parameters.pop("repo_parent_root", None)
    result = main_json(invalid)
    validate_payload(result, ERROR_SCHEMA)
    assert result["status"] == "failure"
