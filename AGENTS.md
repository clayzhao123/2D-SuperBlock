# SuperBlock maintenance instructions

- Read README.md, docs/status.md and docs/architecture.md before modifying code.
- Preserve CLI entry points, callback interfaces and checkpoint payload compatibility.
- Keep learning/environment changes separate from maintenance refactors.
- docs/status.md is the implementation status source of truth; never describe placeholders or plans as completed.
- Do not commit generated experiments, checkpoints, caches or local settings.
- Run relevant regression tests (python -m pytest -q) and a short headless demo. State unavailable tools/GUI checks rather than claiming they passed.
- Verify isolated outputs and recorded failure/interruption for experiment changes.
- Verify failed serialization preserves the previous file for checkpoint changes.
- Report actual validation, changed files and limits.
- Do not add dependencies or directory layers without a concrete need.
