# AGENT.md

## Agent Instructions
- Preserve the existing rich documentation (docstrings and detailed comments) when modifying Python code.  
- Keep all operator-facing documentation in **English** to remain consistent across the repository.  
- After Python changes, always run at least:  
  ```bash
  python -m compileall mailai/src
  ```  
  to check for syntax errors.  

---

## Container & LLM Requirements
- All container workflows must **target `linux/arm64` exclusively**.  
- The MailAI runtime **must** verify during start-up that the embedded `llama-cpp-python` model:  
  1. can be loaded successfully, and  
  2. can produce a completion (warm-up query).  
- Health checks must **fail** if the local LLM cannot serve completions.  

---

## Purpose

This document defines the **rules, standards, and expectations** for developing and maintaining `rpi-mailai`.  
Because this project handles **private email** and applies **automated AI actions**, we enforce **strict documentation, auditability, and safety rules**.  
**No code is merged unless it complies.**

---

## Documentation Standards (Must-Have, Enforced by CI)

We require **exhaustive documentation** across the codebase, optimized for auditability and onboarding.  

### 1) What/Why/How everywhere
- **Module header** at the top of every `.py` file explaining:
  - **What** the module does (responsibilities & public surface)  
  - **Why** it exists (intent, constraints, design trade-offs)  
  - **How** it works (high-level approach, key steps, retries, invariants)  
  - **Interfaces exposed** (public API)  
  - **Invariants & safety rules** (privacy, idempotency, limits, timeouts)  

- **Google-style docstrings** for every **public function/class**, with explicit sections:
  - **What** — behavior, inputs/outputs  
  - **Why** — rationale, constraints (security/perf)  
  - **How** — method/algorithm, key steps, retries, edge cases  
  - Must include arguments, return type(s), and raised exceptions  

- **Private functions (`_name`)** must also have docstrings:
  - **What** (what it does)  
  - **Why** (why this helper exists)  
  - **How** (brief implementation notes)  
  - Arguments/returns may be omitted if trivial, but purpose must be explicit  

- **Inline comments** are reserved for non-trivial logic, prefixed with:
  - `# NOTE:` — subtlety or non-obvious design choice  
  - `# SAFETY:` — security, privacy, or correctness guarantee  
  - `# PERF:` — performance-sensitive part  
  - `# TODO:` — work still required  

### 2) Scope control for large repos
- Document code **incrementally**:  
  - Prefer “first N undocumented modules” per PR (1–5 max).  
  - Stop after N, leave a final `# TODO:` note reminding that others need the same treatment.  

### 3) Typing & linting are part of the contract
- **mypy --strict**: no implicit `Any`, all public APIs typed.  
- **ruff**: lint + docstring rules enabled (`pydocstyle` D-rules).  
- CI **fails** on missing docstrings (even for private helpers) or typing/lint errors.  

---

## PR Requirements (Checklist)

Every PR that adds/changes Python code MUST include:

- [ ] **Module header** present/updated (What/Why/How + invariants).  
- [ ] **Docstrings** for all **public and private** functions/classes added or changed.  
- [ ] **Types** added or updated; `mypy --strict` passes.  
- [ ] **Lint** passes (ruff incl. docstring checks).  
- [ ] If multiple modules are touched, scope ≤ 5 files and add `# TODO` reminders.  
- [ ] Update/add `GUIDE.md` (per directory) if responsibilities/boundaries changed.  
- [ ] CI is green.  

PR template must include this checklist.

---

## Directory-level Guides

Each top-level package (`imap/`, `core/`, `config/`, `storage/`, `utils/`, `cli/`) must include a `GUIDE.md` with:

- **Role (What)**  
- **Intention (Why)**  
- **Approach (How)**  
- **Interfaces & boundaries** with other packages  
- **Invariants** (privacy, idempotency, limits)  
- **Known pitfalls** (IMAP quirks, encoding, timing)  
- **Audit checklist**  

---

## CI Enforcement

Add config in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "D"]
ignore = ["D203", "D213"]  # Google style
exclude = ["build", ".venv"]

[tool.ruff.pydocstyle]
convention = "google"

[tool.mypy]
python_version = "3.11"
strict = true
pretty = true
disallow_any_unimported = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_return_any = true
warn_unused_ignores = true
warn_redundant_casts = true
```

**Pre-commit hooks**:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
```

**CI pipeline**:
- Run ruff (lint+docstring), mypy, pytest.  
- Fail if any docstring, typing, or lint issue.  

---

## Minimal Doc Coverage Test

Add `tests/test_docstrings.py`:

```python
import pkgutil, importlib, inspect, rpi_mailai as root

def test_module_headers():
    for m in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        mod = importlib.import_module(m.name)
        doc = inspect.getdoc(mod) or ""
        assert "What" in doc and "Why" in doc and "How" in doc, f"Missing W/W/H in {m.name}"

def test_function_docstrings():
    for m in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        mod = importlib.import_module(m.name)
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if name.startswith("_") and not inspect.getdoc(obj):
                    raise AssertionError(f"Private {m.name}:{name} missing docstring")
                if not name.startswith("_"):
                    doc = inspect.getdoc(obj) or ""
                    assert "What" in doc and "Why" in doc and "How" in doc, f"Missing W/W/H on {m.name}:{name}"
```

---

## Incremental Workflow

1. **New code** → doc + types in same PR.  
2. **Legacy code modified** → add missing docs for that file.  
3. **Reject PRs** without proper docs unless critical hotfix (must file follow-up doc issue).  

---

## Rationale

This project processes **confidential email data**. We require:
- **Explainability**: every function explains What/Why/How.  
- **Auditability**: auditors can review intent and risks.  
- **Privacy guarantees**: doc must highlight safeguards.  
- **Long-term maintainability**: no “black box” code.  

**Rule:** *No documentation, no merge.*  
