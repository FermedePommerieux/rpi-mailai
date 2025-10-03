# Agent Instructions
- Preserve the existing rich documentation (docstrings and detailed comments) when modifying Python code.
- Keep all operator-facing documentation in English to remain consistent across the repository.
- After Python changes, run at least `python -m compileall mailai/src` to check for syntax errors.

## Container & LLM Requirements
- All container workflows must target `linux/arm64` exclusively.
- The MailAI runtime **must** verify that the embedded `llama-cpp-python` model can load and answer a completion during start-up.
- Health checks must fail if the local LLM cannot serve completions.
