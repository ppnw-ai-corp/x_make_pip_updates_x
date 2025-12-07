from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Final, NamedTuple, Protocol, cast

from x_make_common_x import (
    CommandError,
    isoformat_timestamp,
    log_error,
    log_info,
    run_command,
    write_run_report,
)
from x_make_common_x.json_contracts import validate_payload

from x_make_pip_updates_x.json_contracts import (
    ERROR_SCHEMA,
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
)


class _ValidationErrorProtocol(Protocol):
    message: str
    path: tuple[object, ...]
    schema_path: tuple[object, ...]


class _JsonschemaModule(Protocol):
    ValidationError: type[Exception]


def _load_validation_error() -> type[Exception]:
    module = cast("_JsonschemaModule", importlib.import_module("jsonschema"))
    return module.ValidationError


ValidationErrorType: type[Exception] = _load_validation_error()

_EMPTY_DICT: Final[dict[str, object]] = {}
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType(_EMPTY_DICT)


class PipUpdatesRunnerProtocol(Protocol):
    def batch_install(self, packages: Sequence[str], *, use_user: bool) -> int: ...


class PipUpdatesFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> PipUpdatesRunnerProtocol: ...


class PipUpdatesInstantiationError(RuntimeError):
    """Raised when the pip-updates factory cannot be constructed."""

    def __init__(self) -> None:
        super().__init__(
            "pip updates factory couldn't be instantiated with provided kwargs"
        )


class _NoopRunner(PipUpdatesRunnerProtocol):
    def batch_install(self, packages: Sequence[str], *, use_user: bool) -> int:
        del packages, use_user
        return 0


def _default_runner_factory(
    *_args: object, **_kwargs: object
) -> PipUpdatesRunnerProtocol:
    return _NoopRunner()


def _select_factory(custom_factory: PipUpdatesFactory | None) -> PipUpdatesFactory:
    if custom_factory is not None:
        return custom_factory
    return _default_runner_factory


PACKAGE_ROOT = Path(__file__).resolve().parent


def _info(*parts: object) -> None:
    log_info(*parts)


def _error(*parts: object) -> None:
    log_error(*parts)


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        typed_mapping = cast("Mapping[object, object]", value)
        normalized_mapping: dict[str, object] = {
            str(key): _json_ready(val) for key, val in typed_mapping.items()
        }
        return normalized_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        typed_sequence = cast("Sequence[object]", value)
        normalized_sequence: list[object] = [
            _json_ready(entry) for entry in typed_sequence
        ]
        return normalized_sequence
    return str(value)


def _base_path_from_cloner(cloner: object, repo_parent_root: str) -> Path:
    base_path = Path(repo_parent_root)
    target_attr: object = getattr(cloner, "target_dir", None)
    if isinstance(target_attr, str):
        with suppress(TypeError, ValueError):
            return Path(target_attr)
    if isinstance(target_attr, os.PathLike):
        with suppress(TypeError, ValueError):
            fspath_value = cast("str | bytes", os.fspath(target_attr))
            if isinstance(fspath_value, bytes):
                return Path(fspath_value.decode())
            return Path(fspath_value)
    return base_path


def _resolve_script_path(base_path: Path) -> Path:
    candidates = (
        base_path / "x_4357_make_pip_updates_x" / "x_cls_make_pip_updates_x.py",
        base_path / "x_make_pip_updates_x" / "x_cls_make_pip_updates_x.py",
        Path.cwd() / "x_4357_make_pip_updates_x" / "x_cls_make_pip_updates_x.py",
        Path.cwd() / "x_make_pip_updates_x" / "x_cls_make_pip_updates_x.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_DEFAULT_PACKAGES: tuple[str, ...] = (
    "x_4357_make_markdown_x",
    "x_4357_make_persistent_env_var_x",
    "x_4357_make_pypi_x",
    "x_4357_make_github_clones_x",
    "x_4357_make_pip_updates_x",
)


def _normalize_packages(packages: Sequence[str]) -> list[str]:
    normalized = [pkg for pkg in packages if pkg]
    if normalized:
        deduped: list[str] = []
        seen: set[str] = set()
        for pkg in normalized:
            if pkg not in seen:
                seen.add(pkg)
                deduped.append(pkg)
        return deduped
    return [candidate for candidate in _DEFAULT_PACKAGES if candidate.startswith("x_")]


def _override_use_user_flag(ctx: object | None, *, default: bool) -> bool:
    publish_opts: object | None = None
    if isinstance(ctx, Mapping):
        publish_opts = ctx.get("publish_opts")
    elif ctx is not None:
        publish_opts = getattr(ctx, "publish_opts", None)

    if isinstance(publish_opts, Mapping):
        override = publish_opts.get("use_user")
        if isinstance(override, bool):
            return override
        if isinstance(override, str):
            return override.strip().lower() in {"1", "true", "yes"}
    return default


def _instantiate_runner(
    pip_updates_cls: PipUpdatesFactory,
    *,
    ctx: object | None,
    use_user_flag: bool,
) -> PipUpdatesRunnerProtocol:
    attempt_kwargs: tuple[dict[str, object], ...] = (
        {"user": use_user_flag, "ctx": ctx},
        {"user": use_user_flag},
        {"ctx": ctx},
        {},
    )
    last_exc: TypeError | None = None
    for kwargs in attempt_kwargs:
        try:
            return pip_updates_cls(**kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise PipUpdatesInstantiationError


def _try_run_updates_script(
    pip_updates_cls: PipUpdatesFactory,
    packages: Sequence[str],
    *,
    ctx: object | None,
    use_user_flag: bool,
) -> tuple[bool, int | None]:
    try:
        runner = _instantiate_runner(
            pip_updates_cls,
            ctx=ctx,
            use_user_flag=use_user_flag,
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        _error(
            "pip-updates instantiation failed:",
            f"{exc}; switching to fallback pip install",
        )
        return False, None
    try:
        rc = runner.batch_install(list(packages), use_user=use_user_flag)
    except (
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
        OSError,
    ) as exc:
        _error(
            "pip-updates invocation failed:",
            f"{exc}; switching to fallback pip install",
        )
        return False, None
    if rc != 0:
        _error(
            "pip-updates reported non-zero exit; switching to fallback pip install",
        )
        return True, rc
    return True, rc


def _get_installed_versions(packages: Sequence[str]) -> dict[str, str | None]:
    installed: dict[str, str | None] = {}
    _info("\nInstalled package versions after first update attempt:")
    for pkg in packages:
        try:
            version = importlib_metadata.version(pkg)
            installed[pkg] = version
            _info(f"{pkg}: {version}")
        except importlib_metadata.PackageNotFoundError:
            installed[pkg] = None
            _info(f"{pkg}: not installed (package not found)")
        except ValueError as exc:
            installed[pkg] = None
            _info(f"{pkg}: not installed ({exc})")
    return installed


def _collect_mismatches(
    expected: Mapping[str, str | None],
    installed: Mapping[str, str | None],
) -> list[tuple[str, str, str | None]]:
    mismatches: list[tuple[str, str, str | None]] = []
    for pkg, version in expected.items():
        if not version:
            continue
        current = installed.get(pkg)
        if current != version:
            mismatches.append((pkg, version, current))
    return mismatches


def _retry_mismatches(
    mismatches: Sequence[tuple[str, str, str | None]],
    script_path: Path,
    *,
    use_user_flag: bool,
    final_installed: dict[str, str | None],
) -> int:
    pinned = [f"{pkg}=={version}" for pkg, version, _ in mismatches]
    retry_cmd = [sys.executable, str(script_path)]
    if use_user_flag:
        retry_cmd.append("--user")
    retry_cmd.extend(pinned)
    _info(f"Retrying install for pinned versions: {' '.join(retry_cmd)}")
    retry_proc = run_command(retry_cmd, check=False)
    if retry_proc.stdout:
        _info(retry_proc.stdout.strip())
    if retry_proc.stderr:
        _error(retry_proc.stderr.strip())
    for pkg_name, _, _ in mismatches:
        try:
            final_installed[pkg_name] = importlib_metadata.version(pkg_name)
        except importlib_metadata.PackageNotFoundError:
            final_installed[pkg_name] = None
    return retry_proc.returncode


def _fallback_pip_install(
    packages: Sequence[str],
    published_versions: Mapping[str, str | None],
    *,
    use_user_flag: bool,
) -> None:
    python = sys.executable
    base_cmd = [python, "-m", "pip", "install", "--upgrade"]
    if use_user_flag:
        base_cmd.append("--user")

    pinned = [
        f"{pkg}=={published_versions[pkg]}"
        for pkg in packages
        if published_versions.get(pkg)
    ]
    loose = [pkg for pkg in packages if not published_versions.get(pkg)]

    for batch in (pinned, loose):
        if not batch:
            continue
        cmd = base_cmd + batch
        _info("Fallback pip install:", " ".join(cmd))
        try:
            proc = run_command(cmd, check=True)
        except CommandError as exc:
            _error(str(exc))
            continue
        if proc.stdout:
            _info(proc.stdout.strip())
        if proc.stderr:
            _error(proc.stderr.strip())


def _print_summary(
    published_versions: Mapping[str, str | None],
    final_installed: Mapping[str, str | None],
    packages: Sequence[str],
    retry_rc: int | None,
) -> bool:
    _info("\nSummary:")
    any_failures = False
    iterable: Iterable[tuple[str, str | None]]
    if published_versions:
        iterable = published_versions.items()
    else:
        iterable = ((pkg, None) for pkg in packages)
    for pkg, expected in iterable:
        current = final_installed.get(pkg)
        if expected:
            if current == expected:
                _info(f"- {pkg}=={expected}: OK | current: {current}")
            else:
                any_failures = True
                _info(f"- {pkg}=={expected}: FAIL | current: {current}")
        else:
            _info(f"- {pkg}: current: {current}")
    if retry_rc is not None and retry_rc != 0 and any_failures:
        _error(f"Retry pip-updates failed with exit code {retry_rc}")
    return any_failures


class _UpdateExecutionConfig(NamedTuple):
    pip_updates_factory: PipUpdatesFactory
    ctx: object | None
    use_user_flag: bool
    script_path: Path


def _perform_post_install_verification(
    packages: Sequence[str],
    published_artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _info("Starting post-install verification (lightweight checks only)")
    if not packages:
        _info("No packages provided for verification; skipping")
        return {
            "status": "skipped",
            "reason": "no packages provided",
        }
    missing = [pkg for pkg in packages if pkg not in published_artifacts]
    if missing:
        joined = ", ".join(sorted(missing))
        _info(
            "Skipping detailed verification because artifact metadata is missing for:",
            joined,
        )
        return {
            "status": "skipped",
            "reason": "missing artifact metadata",
            "missing": sorted(missing),
        }
    _info(
        "Detailed file verification is not implemented in the typed orchestrator; "
        "skipping after validating artifact metadata presence",
    )
    return {
        "status": "performed",
        "detail": "metadata validated; deep verification not implemented",
    }


def _prepare_update_execution_details(
    package_list: Sequence[str],
    published_versions: Mapping[str, str | None],
    published_artifacts: Mapping[str, Mapping[str, object]],
    *,
    config: _UpdateExecutionConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    script_exists = config.script_path.is_file()
    used_script = False
    script_rc: int | None = None
    script_detail: dict[str, object]
    if script_exists:
        used_script, script_rc = _try_run_updates_script(
            config.pip_updates_factory,
            package_list,
            ctx=config.ctx,
            use_user_flag=config.use_user_flag,
        )
        script_detail = {
            "invoked": used_script,
            "return_code": script_rc,
        }
    else:
        _info("pip-updates script not found; using direct pip fallback installation")
        script_detail = {
            "invoked": False,
            "return_code": None,
        }

    pinned = [
        f"{pkg}=={published_versions[pkg]}"
        for pkg in package_list
        if published_versions.get(pkg)
    ]
    loose = [pkg for pkg in package_list if not published_versions.get(pkg)]

    used_fallback = (not used_script) or (script_rc not in (None, 0))
    fallback_detail: dict[str, object] = {
        "invoked": used_fallback,
        "pinned": pinned,
        "loose": loose,
    }
    if used_fallback:
        _fallback_pip_install(
            package_list,
            published_versions,
            use_user_flag=config.use_user_flag,
        )

    initial_installed = _get_installed_versions(package_list)
    mismatches = _collect_mismatches(published_versions, initial_installed)
    final_installed = dict(initial_installed)
    retry_rc: int | None = None
    mismatch_entries: list[dict[str, object]] = [
        {
            "package": pkg,
            "expected": expected,
            "observed": observed,
        }
        for pkg, expected, observed in mismatches
    ]

    if mismatches:
        if used_script and not used_fallback and script_exists:
            retry_rc = _retry_mismatches(
                mismatches,
                config.script_path,
                use_user_flag=config.use_user_flag,
                final_installed=final_installed,
            )
            retry_detail: dict[str, object] = {
                "mode": "script",
                "return_code": retry_rc,
                "packages": [entry["package"] for entry in mismatch_entries],
            }
        else:
            _info("Retrying mismatches with pinned fallback pip install")
            _fallback_pip_install(
                [pkg for pkg, _, _ in mismatches],
                published_versions,
                use_user_flag=config.use_user_flag,
            )
            final_installed.update(_get_installed_versions(package_list))
            retry_rc = 0
            retry_detail = {
                "mode": "fallback",
                "return_code": retry_rc,
                "packages": [entry["package"] for entry in mismatch_entries],
            }
        execution_retry = retry_detail
    else:
        execution_retry = {}

    any_failures = _print_summary(
        published_versions,
        final_installed,
        package_list,
        retry_rc,
    )
    verification_detail = _perform_post_install_verification(
        package_list,
        published_artifacts,
    )

    execution_updates: dict[str, object] = {
        "script_attempt": script_detail,
        "fallback": fallback_detail,
    }
    if execution_retry:
        execution_updates["retry"] = execution_retry

    result_updates: dict[str, object] = {
        "status": "completed" if not any_failures else "attention",
        "script_return_code": script_rc,
        "used_script": used_script,
        "fallback_used": used_fallback,
        "retry_return_code": retry_rc,
        "any_failures": any_failures,
        "initial_versions": _json_ready(dict(initial_installed)),
        "final_versions": _json_ready(dict(final_installed)),
        "mismatches": mismatch_entries,
        "verification": verification_detail,
    }
    return execution_updates, result_updates


def run_updates_for_packages(  # noqa: PLR0913
    packages: Sequence[str],
    *,
    cloner: object,
    ctx: object | None,
    repo_parent_root: str,
    published_versions: Mapping[str, str | None],
    published_artifacts: Mapping[str, Mapping[str, object]],
    pip_updates_factory: PipUpdatesFactory,
) -> Path:
    start_time = datetime.now(UTC)
    run_id = uuid.uuid4().hex
    base_path = _base_path_from_cloner(cloner, repo_parent_root)
    script_path = _resolve_script_path(base_path)
    package_list = _normalize_packages(packages)
    use_user_flag = _override_use_user_flag(ctx, default=False)

    inputs_detail: dict[str, object] = {
        "requested_packages": list(packages),
        "normalized_packages": list(package_list),
        "use_user_flag": use_user_flag,
        "repo_parent_root": str(repo_parent_root),
        "published_versions": _json_ready(dict(published_versions)),
        "published_artifacts": _json_ready(dict(published_artifacts)),
    }
    execution_detail: dict[str, object] = {
        "script_path": str(script_path),
        "script_available": script_path.is_file(),
    }
    result_detail: dict[str, object] = {}
    report_payload: dict[str, object] = {
        "run_id": run_id,
        "started_at": isoformat_timestamp(start_time),
        "inputs": inputs_detail,
        "execution": execution_detail,
        "result": result_detail,
    }
    status = "success"
    report_path: Path | None = None
    caught_exc: Exception | None = None
    try:
        if not package_list:
            _info("No published packages to update; skipping pip-updates step")
            result_detail.update(
                {
                    "status": "skipped",
                    "reason": "no packages after normalization",
                }
            )
        else:
            execution_updates, result_updates = _prepare_update_execution_details(
                package_list,
                published_versions,
                published_artifacts,
                config=_UpdateExecutionConfig(
                    pip_updates_factory,
                    ctx,
                    use_user_flag,
                    script_path,
                ),
            )
            execution_detail.update(execution_updates)
            result_detail.update(result_updates)
    except Exception as exc:
        status = "error"
        errors_list: list[dict[str, object]]
        errors_entry = report_payload.get("errors")
        if isinstance(errors_entry, list):
            errors_list = cast("list[dict[str, object]]", errors_entry)
        else:
            errors_list = []
            report_payload["errors"] = errors_list
        errors_list.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        caught_exc = exc
        raise
    finally:
        end_time = datetime.now(UTC)
        report_payload["status"] = status
        report_payload["completed_at"] = isoformat_timestamp(end_time)
        report_payload["duration_seconds"] = round(
            (end_time - start_time).total_seconds(),
            3,
        )
        report_path = write_run_report(
            "x_make_pip_updates_x",
            report_payload,
            base_dir=PACKAGE_ROOT,
        )
        if caught_exc is not None:
            # Preserve the report location for upstream error handling without
            # introducing dynamic attributes that static analyzers reject.
            exc_dict_raw: object | None = getattr(caught_exc, "__dict__", None)
            if isinstance(exc_dict_raw, dict):
                typed_exc_dict = cast("dict[str, object]", exc_dict_raw)
                typed_exc_dict["run_report_path"] = report_path
    return report_path


def _failure_payload(
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "failure",
        "message": message,
    }
    if details:
        payload["details"] = {str(key): value for key, value in details.items()}
    with suppress(ValidationErrorType):
        validate_payload(payload, ERROR_SCHEMA)
    return payload


def _missing_report_payload() -> dict[str, object]:
    return _failure_payload("pip updates report missing", details=None)


def _parameters_from_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    parameters_obj: object | None = payload.get("parameters")
    if isinstance(parameters_obj, Mapping):
        typed_parameters = cast("Mapping[str, object]", parameters_obj)
        sanitized_parameters: dict[str, object] = {
            str(key): value for key, value in typed_parameters.items()
        }
        return MappingProxyType(sanitized_parameters)
    return _EMPTY_MAPPING


def _coerce_packages(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(entry) for entry in value if str(entry))
    return ()


def _coerce_published_versions(value: object) -> dict[str, str | None]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str | None] = {}
    for key, entry in value.items():
        coerced: str | None = entry if isinstance(entry, str) and entry else None
        result[str(key)] = coerced
    return result


def _coerce_published_artifacts(value: object) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for key, entry in value.items():
        if isinstance(entry, Mapping):
            typed_entry = cast("Mapping[object, object]", entry)
            sanitized: dict[str, object] = {
                str(sub_key): sub_value for sub_key, sub_value in typed_entry.items()
            }
            result[str(key)] = sanitized
    return result


def _resolve_effective_ctx(
    ctx: object | None,
    context_obj: object,
) -> object | None:
    if ctx is not None:
        return ctx
    if isinstance(context_obj, Mapping):
        sanitized: dict[str, object] = {
            str(key): value for key, value in context_obj.items()
        }
        return sanitized
    return None


def _build_cloner(source: object) -> SimpleNamespace:
    if isinstance(source, Mapping):
        sanitized: dict[str, object] = {
            key: value
            for key, value in source.items()
            if isinstance(key, str) and key.isidentifier()
        }
        return SimpleNamespace(**sanitized)
    return SimpleNamespace()


@dataclass(frozen=True)
class _UpdatesStageParams:
    cloner: SimpleNamespace
    effective_ctx: object | None
    repo_parent_root: str
    published_versions: Mapping[str, str | None]
    published_artifacts: Mapping[str, Mapping[str, object]]
    factory: PipUpdatesFactory


class _PipelineError(RuntimeError):
    """Raised when any step in the pip updates pipeline fails."""

    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__("pip updates pipeline failed")
        self.payload = payload


def _ensure_success(payload: dict[str, object] | None) -> None:
    if payload is not None:
        raise _PipelineError(payload)


def _validate_input_payload(payload: Mapping[str, object]) -> dict[str, object] | None:
    try:
        validate_payload(payload, INPUT_SCHEMA)
    except ValidationErrorType as exc:
        error = cast("_ValidationErrorProtocol", exc)
        return _failure_payload(
            "input payload failed validation",
            details={
                "error": error.message,
                "path": [str(part) for part in error.path],
                "schema_path": [str(part) for part in error.schema_path],
            },
        )
    return None


def _require_repo_parent_root(
    parameters: Mapping[str, object],
) -> tuple[str | None, dict[str, object] | None]:
    candidate = parameters.get("repo_parent_root")
    if isinstance(candidate, str) and candidate:
        return candidate, None
    return None, _failure_payload(
        "repo_parent_root is required",
        details={"field": "repo_parent_root"},
    )


def _run_updates_stage(
    packages: Sequence[str],
    *,
    params: _UpdatesStageParams,
) -> tuple[Path | None, dict[str, object] | None]:
    try:
        report_path = run_updates_for_packages(
            packages,
            cloner=params.cloner,
            ctx=params.effective_ctx,
            repo_parent_root=params.repo_parent_root,
            published_versions=params.published_versions,
            published_artifacts=params.published_artifacts,
            pip_updates_factory=params.factory,
        )
    except Exception as exc:  # noqa: BLE001 - convert to schema failure payload
        return None, _failure_payload(
            "pip updates execution failed",
            details={
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
    return report_path, None


def _load_report_payload(
    report_path: Path,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            loaded_obj: object = json.load(handle)
    except FileNotFoundError:
        return None, _failure_payload(
            "pip updates report not found",
            details={"report_path": str(report_path)},
        )
    if not isinstance(loaded_obj, Mapping):
        message = "pip updates report must be a mapping"
        return None, _failure_payload(
            "pip updates report malformed",
            details={"message": message},
        )
    typed_loaded = cast("Mapping[str, object]", loaded_obj)
    return dict(typed_loaded), None


def _validate_output_payload(
    result_payload: Mapping[str, object],
) -> dict[str, object] | None:
    try:
        validate_payload(result_payload, OUTPUT_SCHEMA)
    except ValidationErrorType as exc:
        error = cast("_ValidationErrorProtocol", exc)
        return _failure_payload(
            "generated output failed schema validation",
            details={
                "error": error.message,
                "path": [str(part) for part in error.path],
                "schema_path": [str(part) for part in error.schema_path],
            },
        )
    return None


def main_json(
    payload: Mapping[str, object],
    *,
    ctx: object | None = None,
    pip_updates_factory: PipUpdatesFactory | None = None,
) -> dict[str, object]:
    """Execute the pip updates flow using the JSON contract."""

    try:
        return _main_json_impl(
            payload,
            ctx=ctx,
            pip_updates_factory=pip_updates_factory,
        )
    except _PipelineError as exc:
        return exc.payload


def _main_json_impl(
    payload: Mapping[str, object],
    *,
    ctx: object | None,
    pip_updates_factory: PipUpdatesFactory | None,
) -> dict[str, object]:
    error_response = _validate_input_payload(payload)
    _ensure_success(error_response)

    parameters = _parameters_from_payload(payload)
    packages = _coerce_packages(parameters.get("packages"))

    repo_parent_root, repo_error = _require_repo_parent_root(parameters)
    _ensure_success(repo_error)
    if repo_parent_root is None:
        raise _PipelineError(
            _failure_payload(
                "repo_parent_root is required",
                details={"field": "repo_parent_root"},
            )
        )

    published_versions = _coerce_published_versions(
        parameters.get("published_versions")
    )
    published_artifacts = _coerce_published_artifacts(
        parameters.get("published_artifacts")
    )

    context_obj = parameters.get("context")
    effective_ctx = _resolve_effective_ctx(ctx, context_obj)
    cloner_obj = _build_cloner(parameters.get("cloner"))

    factory = _select_factory(pip_updates_factory)
    stage_params = _UpdatesStageParams(
        cloner=cloner_obj,
        effective_ctx=effective_ctx,
        repo_parent_root=repo_parent_root,
        published_versions=published_versions,
        published_artifacts=published_artifacts,
        factory=factory,
    )

    report_path, stage_error = _run_updates_stage(
        packages,
        params=stage_params,
    )
    _ensure_success(stage_error)

    if report_path is None:
        raise _PipelineError(_missing_report_payload())

    result_payload, load_error = _load_report_payload(report_path)
    _ensure_success(load_error)
    if result_payload is None:
        raise _PipelineError(_missing_report_payload())

    output_validation_error = _validate_output_payload(result_payload)
    _ensure_success(output_validation_error)

    return result_payload


__all__ = [
    "PipUpdatesFactory",
    "PipUpdatesRunnerProtocol",
    "main_json",
    "run_updates_for_packages",
]
