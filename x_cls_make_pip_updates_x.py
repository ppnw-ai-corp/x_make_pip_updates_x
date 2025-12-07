from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from pathlib import Path
from types import MappingProxyType
from typing import IO, cast

from x_make_common_x import CommandError, run_command

from x_make_pip_updates_x.update_flow import main_json

_LOGGER = logging.getLogger("x_make")
_sys = sys
PACKAGE_ROOT = Path(__file__).resolve().parent


def _info(*args: object) -> None:
    msg = " ".join(str(a) for a in args)
    with suppress(Exception):
        _LOGGER.info("%s", msg)
    printed = False
    with suppress(Exception):
        print(msg)
        printed = True
    if not printed:
        with suppress(Exception):
            _sys.stdout.write(msg + "\n")


def _error(*args: object) -> None:
    msg = " ".join(str(a) for a in args)
    with suppress(Exception):
        _LOGGER.error("%s", msg)
    wrote = False
    with suppress(Exception):
        print(msg, file=_sys.stderr)
        wrote = True
    if not wrote:
        with suppress(Exception):
            _sys.stderr.write(msg + "\n")
            wrote = True
    if not wrote:
        with suppress(Exception):
            print(msg)


"""red rabbit 2025_0902_0944"""


# use shared helpers from x_make_common_x.helpers


RunResult = tuple[int, str, str]


@dataclass(slots=True)
class InstallResult:
    name: str
    prev: str | None
    curr: str | None
    code: int


class PipUpdatesRunner:
    # ...existing code...

    @staticmethod
    def _ctx_flag(ctx: object | None, attr: str) -> bool:
        if ctx is None:
            return False
        if isinstance(ctx, Mapping):
            mapping_ctx = cast("Mapping[str, object]", ctx)
            value = mapping_ctx.get(attr, False)
        else:
            try:
                value = cast("object", getattr(ctx, attr, False))
            except AttributeError:
                return False
        raw: object = value
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return raw != 0
        if isinstance(raw, str):
            return raw.lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    def batch_install(self, packages: Sequence[str], *, use_user: bool = False) -> int:
        # Force pip upgrade first
        _info("Upgrading pip itself...")
        pip_upgrade_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
        ]
        pip_upgrade_code = self._run_and_report(pip_upgrade_cmd)[0]
        if pip_upgrade_code != 0:
            _info("Failed to upgrade pip. Continuing anyway.")

        seen: set[str] = set()
        normalized: list[str] = []
        for pkg in packages:
            if pkg not in seen:
                seen.add(pkg)
                normalized.append(pkg)
        if not normalized:
            _info("No packages supplied; nothing to do.")
            return 0

        _info(
            "Upgrading all published packages with "
            "--upgrade --force-reinstall --no-cache-dir..."
        )

        results = [self._refresh_package(pkg, use_user=use_user) for pkg in normalized]
        return self._summarize(results)

    """
    Ensure a Python package is installed and up-to-date in the current interpreter.

    - Installs the package if missing.
    - Upgrades only when the installed version is outdated.
    - Uses the same Python executable (sys.executable -m pip).
    """

    def __init__(self, *, user: bool = False, ctx: object | None = None) -> None:
        """Primary constructor: preserve previous 'user' flag and accept ctx.

        Dry-run is now sourced from the orchestrator context when provided.
        If no context is provided, default to False.
        """
        self.user = user
        self._ctx = ctx
        self.dry_run = self._ctx_flag(self._ctx, "dry_run")

        if self._ctx_flag(self._ctx, "verbose"):
            _info(f"[pip_updates] initialized user={self.user}")

    def _run_and_report(self, cmd: Sequence[str], *, check: bool = True) -> RunResult:
        try:
            result = run_command(list(cmd), check=check)
        except CommandError as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if stdout.strip():
                _info(stdout.strip())
            if stderr.strip():
                _error(stderr.strip())
            return exc.returncode, stdout, stderr

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if stdout.strip():
            _info(stdout.strip())
        if stderr.strip() and result.returncode != 0:
            _error(stderr.strip())
        return result.returncode, stdout, stderr

    @staticmethod
    def _run(cmd: Sequence[str]) -> RunResult:
        completed = run_command(cmd, check=False)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        return completed.returncode, stdout, stderr

    @staticmethod
    def get_installed_version(dist_name: str) -> str | None:
        try:
            res = _version(dist_name)
            return str(res)
        except PackageNotFoundError:
            return None
        except ValueError as exc:  # pragma: no cover - defensive logging
            _error(f"Failed to query version for {dist_name}: {exc}")
            return None

    def is_outdated(self, dist_name: str) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "list",
            "--outdated",
            "--format=json",
            "--disable-pip-version-check",
        ]
        code, out, err = self._run(cmd)
        if code != 0:
            _error(f"pip list failed ({code}): {err.strip()}")
            return False
        try:
            decoded: object = json.loads(out or "[]")
        except json.JSONDecodeError:
            return False

        if not isinstance(decoded, list):
            return False

        decoded_list = cast("list[object]", decoded)
        for entry_obj in decoded_list:
            if not isinstance(entry_obj, dict):
                continue
            entry_mapping = cast("dict[object, object]", entry_obj)
            entry: dict[str, object] = {
                key_obj: value_obj
                for key_obj, value_obj in entry_mapping.items()
                if isinstance(key_obj, str)
            }
            name_obj = entry.get("name")
            if isinstance(name_obj, str) and name_obj.lower() == dist_name.lower():
                return True
        return False

    def pip_install(self, dist_name: str, *, upgrade: bool = False) -> int:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
        ]
        if upgrade:
            cmd.append("--upgrade")
        if self.user:
            cmd.append("--user")
        cmd.append(dist_name)
        return self._run_and_report(cmd)[0]

    def ensure(self, dist_name: str) -> None:
        installed = self.get_installed_version(dist_name)
        if installed is None:
            _info(f"{dist_name} not installed. Installing...")
            code = self.pip_install(dist_name, upgrade=False)
            if code != 0:
                _error(f"Failed to install {dist_name} (exit {code}).")
            return
        _info(f"{dist_name} installed (version {installed}). Checking for updates...")
        if self.is_outdated(dist_name):
            _info(f"{dist_name} is outdated. Upgrading...")
            code = self.pip_install(dist_name, upgrade=True)
            if code != 0:
                _error(f"Failed to upgrade {dist_name} (exit {code}).")
        else:
            _info(f"{dist_name} is up to date.")

    def _refresh_package(self, package: str, *, use_user: bool) -> InstallResult:
        previous = self.get_installed_version(package)
        self.user = use_user
        cmd = self._build_refresh_command(package=package, use_user=use_user)
        code = self._run_and_report(cmd)[0]
        current = self.get_installed_version(package)
        return InstallResult(package, previous, current, code)

    @staticmethod
    def _build_refresh_command(*, package: str, use_user: bool) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
        ]
        if use_user:
            cmd.append("--user")
        cmd.append(package)
        return cmd

    def _summarize(self, results: Iterable[InstallResult]) -> int:
        entries = list(results)
        if not entries:
            _info("No packages were processed.")
            return 0

        _info("\nSummary:")
        any_fail = False
        for result in entries:
            prev = result.prev or "not installed"
            curr = result.curr or "not installed"
            status = "OK" if result.code == 0 else f"FAIL (code {result.code})"
            if result.code != 0:
                any_fail = True
            _info(f"- {result.name}: {status} | previous: {prev} | current: {curr}")
        return 1 if any_fail else 0


x_cls_make_pip_updates_x = PipUpdatesRunner


def _load_json_payload(file_path: str | None) -> Mapping[str, object]:
    def _load(stream: IO[str]) -> Mapping[str, object]:
        payload_obj: object = json.load(stream)
        if not isinstance(payload_obj, Mapping):
            message = "JSON payload must be a mapping"
            raise TypeError(message)
        typed_payload = cast("Mapping[object, object]", payload_obj)
        sanitized: dict[str, object] = {
            key: value for key, value in typed_payload.items() if isinstance(key, str)
        }
        return MappingProxyType(sanitized)

    if file_path:
        with Path(file_path).open("r", encoding="utf-8") as handle:
            return _load(handle)
    return _load(sys.stdin)


def _run_json_cli(args: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="x_make_pip_updates_x JSON runner")
    parser.add_argument(
        "--json", action="store_true", help="Read JSON payload from stdin"
    )
    parser.add_argument("--json-file", type=str, help="Path to JSON payload file")
    parsed: argparse.Namespace = parser.parse_args(args)
    json_attr = cast("object", getattr(parsed, "json", False))
    json_flag = bool(json_attr)
    json_file_attr = cast("object", getattr(parsed, "json_file", None))
    json_file: str | None = json_file_attr if isinstance(json_file_attr, str) else None

    if not (json_flag or json_file):
        parser.error("JSON input required. Use --json for stdin or --json-file <path>.")

    payload = _load_json_payload(json_file)
    result = main_json(payload)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _run_json_cli(sys.argv[1:])
