"""Minimal YAML loader/dumper with optional PyYAML integration."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


try:  # pragma: no cover - exercised when PyYAML is available
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback executed in tests
    yaml = None


def load(source: bytes) -> Any:
    if yaml is not None:
        return yaml.safe_load(source)
    return _parse_simple_yaml(source.decode("utf-8"))


def dump(data: Any) -> str:
    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)
    return _dump_simple_yaml(data)


def _parse_simple_yaml(text: str) -> Any:
    lines = text.splitlines()
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed


def _parse_block(lines: List[str], indent: int, index: int) -> Tuple[Any, int]:
    block_lines = []
    idx = index
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            block_lines.append(line)
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        block_lines.append(line)
        idx += 1
    if not block_lines:
        return {}, idx
    for line in block_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            return _parse_list(block_lines, indent), idx
        break
    return _parse_mapping(block_lines, indent), idx


def _parse_mapping(lines: List[str], indent: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if stripped.startswith("- "):
            raise ValueError("Unexpected list item in mapping context")
        key, _, value = stripped.partition(":")
        if not _:
            raise ValueError("Invalid mapping entry")
        idx += 1
        if value.strip() == "":
            sub_lines: List[str] = []
            while idx < len(lines):
                next_line = lines[idx]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped.startswith("#"):
                    sub_lines.append(next_line)
                    idx += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= current_indent:
                    break
                sub_lines.append(next_line)
                idx += 1
            value_parsed, _ = _parse_block(sub_lines, current_indent + 2, 0)
            result[key] = value_parsed
        else:
            result[key] = _parse_scalar(value.strip())
    return result


def _parse_list(lines: List[str], indent: int) -> List[Any]:
    items: List[Any] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            idx += 1
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if not stripped.startswith("- "):
            raise ValueError("Invalid list entry")
        item_lines: List[str] = [line]
        idx += 1
        while idx < len(lines):
            next_line = lines[idx]
            next_stripped = next_line.strip()
            if not next_stripped or next_stripped.startswith("#"):
                item_lines.append(next_line)
                idx += 1
                continue
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= current_indent:
                break
            item_lines.append(next_line)
            idx += 1
        value = _parse_list_item(item_lines, current_indent)
        items.append(value)
    return items


def _parse_list_item(lines: List[str], indent: int) -> Any:
    first = lines[0]
    value_part = first.strip()[2:]
    nested_lines = lines[1:]
    if value_part == "" and not nested_lines:
        return None
    if value_part == "":
        parsed, _ = _parse_block(nested_lines, indent + 2, 0)
        return parsed
    if _has_unquoted_colon(value_part):
        inline = " " * (indent + 2) + value_part
        block_lines = [inline] + nested_lines
        parsed, _ = _parse_block(block_lines, indent + 2, 0)
        return parsed
    if nested_lines:
        block_lines = [" " * (indent + 2) + value_part] + nested_lines
        parsed, _ = _parse_block(block_lines, indent + 2, 0)
        return parsed
    return _parse_scalar(value_part.strip())


def _has_unquoted_colon(text: str) -> bool:
    stripped = text.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return False
    return ":" in text


def _parse_scalar(token: str) -> Any:
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if token.lower() == "null":
        return None
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _dump_simple_yaml(data: Any, indent: int = 0) -> str:
    space = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_dump_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{space}{key}: {_format_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{space}- {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{space}{_format_scalar(data)}"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    text = str(value)
    if any(ch in text for ch in [":", "#", "\n", "\\"]):
        return f'"{text}"'
    return text
