# Agent Instructions

- Preserve the existing rich documentation (docstrings and detailed comments) when modifying Python code.
- Keep all operator-facing documentation in English to remain consistent across the repository.
- After Python changes, run at least `python -m compileall app` to check for syntax errors.

## Container & LLM Requirements
- All container workflows must target `linux/arm64` exclusively.
- The MailAI runtime **must** verify that an LLM endpoint is reachable during start-up and refuse to run otherwise.
- Health checks must fail if the LLM cannot serve chat completions.
