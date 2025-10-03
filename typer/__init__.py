"""Minimal Typer compatibility layer for environments without the dependency.

This shim implements the subset of the Typer API required by the tests in this
exercise. It should not be considered a drop-in replacement for Typer in real
deployments.
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Optional, Union, get_args, get_origin, get_type_hints


class Exit(SystemExit):
    """Exception raised to exit the CLI with a specific status code."""

    def __init__(self, code: int = 0) -> None:
        super().__init__(code)
        self.code = code


def Option(default: Any, *, help: str | None = None) -> Any:
    """Return ``default`` to emulate Typer's ``Option`` helper."""

    return default


def Argument(default: Any, *, help: str | None = None) -> Any:
    """Return ``default`` mirroring Typer's ``Argument`` helper."""

    return default


def _convert(value: str, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if annotation in (int, Optional[int]):
        return int(value)
    if origin in (Union, Optional):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return _convert(value, args[0])
    return value


def _parse_options(func: Callable[..., Any], arguments: Iterable[str]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    values = {name: parameter.default for name, parameter in signature.parameters.items()}
    tokens = list(arguments)
    index = 0
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    positional_index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            if positional_index >= len(positional):
                raise Exit(code=1)
            parameter = positional[positional_index]
            annotation = hints.get(parameter.name, parameter.annotation)
            values[parameter.name] = _convert(token, annotation)
            positional_index += 1
            index += 1
            continue
        name = token[2:].replace("-", "_")
        if name not in signature.parameters:
            raise Exit(code=1)
        index += 1
        if index >= len(tokens):
            raise Exit(code=1)
        raw_value = tokens[index]
        parameter = signature.parameters[name]
        annotation = hints.get(name, parameter.annotation)
        values[name] = _convert(raw_value, annotation)
        index += 1
    return values


class Typer:
    """Simplified Typer application supporting command registration."""

    def __init__(self, *, help: str | None = None) -> None:
        self._help = help
        self._commands: Dict[str, Callable[..., Any]] = {}

    def command(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            command_name = name or func.__name__.replace("_", "-")
            self._commands[command_name] = func
            return func

        return decorator

    def __call__(self) -> None:
        args = sys.argv[1:]
        if not args:
            raise Exit(code=1)
        command_name, *rest = args
        handler = self._commands.get(command_name)
        if handler is None:
            raise Exit(code=1)
        kwargs = _parse_options(handler, rest)
        handler(**kwargs)


@dataclass
class _Result:
    exit_code: int
    output: str = ""
    exception: Exception | None = None


class CliRunner:
    """Minimal clone of Typer's ``CliRunner`` for unit tests."""

    def invoke(self, app: Typer, args: Iterable[str]) -> _Result:
        try:
            command_name, *rest = args
        except ValueError:
            return _Result(exit_code=1)
        handler = app._commands.get(command_name)
        if handler is None:
            return _Result(exit_code=1)
        try:
            kwargs = _parse_options(handler, rest)
            handler(**kwargs)
        except Exit as exc:
            return _Result(exit_code=exc.code)
        except Exception as exc:  # pragma: no cover - best-effort shim
            return _Result(exit_code=1, exception=exc)
        return _Result(exit_code=0)


testing_module = ModuleType("typer.testing")
testing_module.CliRunner = CliRunner
sys.modules.setdefault("typer.testing", testing_module)

