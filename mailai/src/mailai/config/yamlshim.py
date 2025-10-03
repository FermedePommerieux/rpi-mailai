"""YAML serialization shim with graceful PyYAML fallback.

What:
  Expose ``load`` and ``dump`` helpers that prefer ``PyYAML`` for full-featured
  parsing but fall back to a constrained parser capable of handling the limited
  configuration syntax used by MailAI.

Why:
  Raspberry Pi deployments ship minimal environments; relying exclusively on
  ``PyYAML`` would complicate installation. The shim guarantees deterministic
  behaviour even when optional dependencies are absent while retaining the same
  interface surface.

How:
  Attempts to import ``yaml`` at module import time. When available, delegates
  to ``safe_load``/``safe_dump``. Otherwise a handcrafted recursive-descent
  parser processes mappings and lists with predictable indentation handling and
  scalar coercion suitable for configuration files.

Interfaces:
  ``load`` and ``dump`` functions.

Invariants & Safety:
  - Only a subset of YAML 1.2 is supported in fallback mode (mappings, lists,
    scalars, and comments); this is sufficient for MailAI configs.
  - Parsed scalars never execute arbitrary code; no object constructors are
    supported.
  - Indentation is restricted to spaces, matching MailAI's linted examples.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


try:  # pragma: no cover - exercised when PyYAML is available
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - fallback executed in tests
    yaml = None


def load(source: bytes) -> Any:
    """Deserialize YAML payloads.

    What:
      Converts raw bytes into Python primitives matching MailAI's configuration
      schema.

    Why:
      Configurations are delivered through email bodies and must be parsed on
      constrained devices without optional dependencies.

    How:
      Delegates to :mod:`PyYAML` when present; otherwise decodes UTF-8 bytes and
      feeds them to the minimal parser.

    Args:
      source: YAML payload encoded as bytes.

    Returns:
      Parsed Python representation (dicts, lists, scalars).

    Raises:
      ValueError: If fallback parsing encounters unsupported structures.
    """

    if yaml is not None:
        return yaml.safe_load(source)
    return _parse_simple_yaml(source.decode("utf-8"))


def dump(data: Any) -> str:
    """Serialize Python structures into YAML text.

    What:
      Produces a human-readable YAML string compatible with MailAI examples and
      documentation.

    Why:
      Enables emitting configuration templates, diagnostics, and backups even on
      hosts without ``PyYAML``.

    How:
      Uses ``yaml.safe_dump`` when available. The fallback mirrors ``load`` by
      traversing dictionaries and lists recursively, ensuring deterministic
      ordering consistent with how MailAI generates samples.

    Args:
      data: Python structure to serialize.

    Returns:
      YAML string without trailing newline to align with existing expectations.
    """

    if yaml is not None:
        return yaml.safe_dump(data, sort_keys=False)
    return _dump_simple_yaml(data)


def _parse_simple_yaml(text: str) -> Any:
    """Parse a limited subset of YAML into Python primitives.

    What:
      Handle mappings and lists composed of scalars, mirroring the minimal
      syntax accepted by MailAI configuration emails.

    Why:
      Raspberry Pi deployments may lack PyYAML; the shim provides a predictable
      fallback parser covering the project's constrained YAML dialect.

    How:
      Split ``text`` into lines and delegate to :func:`_parse_block`, which
      recurses through mappings and lists based on indentation.

    Args:
      text: YAML string to parse.

    Returns:
      Python object representing the parsed YAML structure.
    """

    lines = text.splitlines()
    parsed, _ = _parse_block(lines, 0, 0)
    return parsed


def _parse_block(lines: List[str], indent: int, index: int) -> Tuple[Any, int]:
    """Parse a block of YAML lines starting at ``index``.

    What:
      Determine whether the block represents a mapping or list and parse the
      structure accordingly.

    Why:
      The shim relies on indentation to infer structure; centralising the logic
      ensures consistent treatment of blank lines and comments.

    How:
      Collect contiguous lines with indentation greater than or equal to
      ``indent`` and dispatch to :func:`_parse_list` or :func:`_parse_mapping`
      based on the first meaningful token.

    Args:
      lines: All input lines.
      indent: Current indentation depth.
      index: Starting line index.

    Returns:
      Tuple of (parsed object, next index).
    """

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
        if stripped == "{}":
            return {}, idx
        if stripped == "[]":
            return [], idx
        if stripped.startswith("-"):
            return _parse_list(block_lines, indent), idx
        break
    return _parse_mapping(block_lines, indent), idx


def _parse_mapping(lines: List[str], indent: int) -> Dict[str, Any]:
    """Parse a mapping block into a dictionary.

    What:
      Interpret ``key: value`` pairs, supporting nested blocks when values are
      indented on subsequent lines.

    Why:
      MailAI configurations lean on mappings for readability; the shim must
      support nested dictionaries without external dependencies.

    How:
      Iterate through ``lines``, extract keys, and recursively parse nested
      blocks using :func:`_parse_block` when values are omitted inline.

    Args:
      lines: Lines belonging to the mapping.
      indent: Expected indentation of the mapping entries.

    Returns:
      Dictionary representing the parsed mapping.
    """

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
            if not any(line.strip() for line in sub_lines):
                result[key] = ""
            else:
                value_parsed, _ = _parse_block(sub_lines, current_indent + 2, 0)
                result[key] = value_parsed
        else:
            result[key] = _parse_scalar(value.strip())
    return result


def _parse_list(lines: List[str], indent: int) -> List[Any]:
    """Parse a sequence block into a list of Python objects.

    What:
      Interpret ``- item`` entries, allowing nested structures under each item.

    Why:
      Lists capture ordered rules and actions; the shim needs to preserve order
      while supporting nested mappings or lists.

    How:
      Iterate through list entries, collect their indented continuations, and
      parse each item via :func:`_parse_list_item`.

    Args:
      lines: Lines comprising the list.
      indent: Indentation level of the list marker.

    Returns:
      List containing parsed values for each entry.
    """

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
        if not stripped.startswith("-"):
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
    """Parse a single ``-`` list entry, handling nested content.

    What:
      Interpret the inline value following the dash or recursively parse nested
      blocks when the entry spans multiple lines.

    Why:
      Many configuration lists embed mappings; this helper ensures indentation
      driven parsing remains robust.

    How:
      Examine the first line after the dash; when empty, treat subsequent lines
      as a nested block passed to :func:`_parse_block`, otherwise parse the
      scalar inline.

    Args:
      lines: Lines forming the list item (starting with ``-``).
      indent: Base indentation of the list item.

    Returns:
      Parsed Python object representing the list entry.
    """

    first = lines[0]
    stripped = first.strip()
    value_part = stripped[2:] if len(stripped) > 1 else ""
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
    """Return ``True`` when ``text`` contains an unquoted colon.

    What:
      Detect potential mapping delimiters to decide whether quoting is required
      during serialisation.

    Why:
      Colons outside quotes would be misinterpreted as mapping separators when
      dumping YAML, so scalars containing them must be quoted.

    How:
      Strip surrounding whitespace, ignore text already wrapped in matching
      quotes, and check for a ``:`` character.

    Args:
      text: Scalar string candidate.

    Returns:
      ``True`` if quoting is necessary, otherwise ``False``.
    """

    stripped = text.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return False
    return ":" in text


def _parse_scalar(token: str) -> Any:
    """Interpret a scalar token into Python primitives.

    What:
      Convert quoted strings, booleans, numbers, and ``null`` into their Python
      equivalents.

    Why:
      Ensures textual representations from YAML map to canonical Python types
      that downstream code expects.

    How:
      Inspect quoting, look for boolean/null literals, attempt numeric casts, and
      fall back to the raw string when coercion fails.

    Args:
      token: Scalar string extracted from YAML.

    Returns:
      Parsed Python value.
    """

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
    """Serialise Python structures into the restricted YAML dialect.

    What:
      Convert dictionaries, lists, and scalars into YAML text matching the
      parser's expectations.

    Why:
      Round-tripping status documents requires deterministic formatting even
      when PyYAML is unavailable.

    How:
      Recursively process mappings and lists, indenting nested structures and
      delegating scalar formatting to :func:`_format_scalar`.

    Args:
      data: Object to serialise.
      indent: Current indentation level.

    Returns:
      YAML string representing ``data``.
    """

    space = " " * indent
    if isinstance(data, dict):
        if not data:
            return f"{space}{{}}"
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_dump_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{space}{key}: {_format_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        if not data:
            return f"{space}[]"
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
    """Render ``value`` as a YAML-safe scalar string.

    What:
      Convert Python primitives into textual representation suitable for the
      shim's limited YAML emitter.

    Why:
      Ensures special characters and reserved literals are quoted appropriately
      so round-tripped documents remain parseable.

    How:
      Inspect the value type, format booleans/literals accordingly, and quote
      strings containing problematic characters.

    Args:
      value: Scalar to format.

    Returns:
      String representation safe for inclusion in YAML.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in [":", "#", "\n", "\\"]):
        return f'"{text}"'
    return text


# TODO: Document remaining modules with the same What/Why/How structure.
