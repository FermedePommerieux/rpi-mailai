"""Test package initialiser documenting shared fixtures and pytest setup.

What:
  Marks ``tests`` as a package so pytest can import reusable fixtures from nested
  modules.

Why:
  Some tooling expects a package structure when referencing ``tests.unit`` or
  ``tests.e2e`` modules explicitly. Providing the marker file keeps imports
  deterministic.

How:
  The file intentionally exposes no symbols; it merely explains why the package
  exists and how the directory hierarchy is consumed by pytest.

Interfaces:
  No public interfaces are defined here.

Invariants & Safety:
  - The file must remain side-effect free so that importing ``tests`` never
    mutates environment state or test fixtures.
"""


# TODO: Other modules in this repository still require the same What/Why/How documentation.