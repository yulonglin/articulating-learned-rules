# Repository Guidelines

## Project Structure & Module Organization
`src/` holds the core Python modules, including the async caller (`api_caller.py`), experiment orchestration (`runner.py`), and evaluation scripts. Model-specific analyses stay beside their focused tests (`test_*.py`) so fixtures remain local. `experiments/` stores timestamped run outputs plus the shared `registry.yaml`; archive or reference large artifacts instead of committing them. Use `tmp/` for planning templates, keep `examples/` for published prompts/datasets, and edit existing files rather than introducing new docs unless a maintainer approves it.

## Build, Test, and Development Commands
- `uv pip install -e .` installs the package and extras into your active environment.
- `uv run python -m src.runner --help` lists all experiment CLI flags; clone this call when starting a new run.
- `uv run python -m src.runner --experiment-name 20250201_my_run --output-dir experiments/20250201_my_run --model gpt-4o-mini --cache-mode short` is the standard launch template; adjust arguments instead of editing code.
- `uv run pytest src` executes the full test suite; append a path (e.g., `src/test_learnability.py::test_precision`) for targeted runs.

## Coding Style & Naming Conventions
Use Python 3.9+ with four-space indentation, descriptive snake_case for modules/functions, and CapWords for classes. Maintain full type hints and docstrings as shown in `src/utils.py`. Keep imports at the top of each file, let errors propagate (no sweeping try/except), and avoid fallback logic that could mask regressions. Run `uv run ruff check src` and `uv run mypy src` before submitting changes; resolve every warning instead of muting it.

## Testing Guidelines
Author tests with `pytest`, mirroring the `test_<feature>.py` pattern already in `src/`. Tests should assert exact behaviors (e.g., cached responses, registry updates) and use real data paths, not mocks, unless explicitly documented. Aim to cover new branches and failure modes added by your change; rerun `uv run pytest src` before opening a pull request. Capture relevant seeds or fixtures inline so failures are reproducible.

## Commit & Pull Request Guidelines
Follow an imperative, present-tense commit style such as `Add learnability aggregation helper`; keep subjects ≤72 characters, never mention “Claude,” and add concise bodies when context is non-obvious. Commits should stay focused on one concern (code, docs, or experiments) and avoid mixing incidental formatting. Pull requests must describe the motivation, summarize key changes, link to any experiment artifacts in `experiments/`, and call out testing performed (commands plus results). Include screenshots only when UI output or plots change; otherwise attach paths to generated figures.

## Security & Configuration Tips
Populate secrets in `.env` locally and ensure `.cache/` stays gitignored. Prefer `CacheMode.SHORT` while iterating and switch to `CacheMode.PERSISTENT` only for final runs. Update `experiments/registry.yaml` immediately after each experiment and double-check that output directories use the `YYYYMMDD_experiment` naming pattern. Never commit API responses containing sensitive data; reference them by path instead.

## Operational Guardrails
Do not delete existing repository content unless explicitly instructed, and never run destructive commands. Avoid creating new files by default—modify current artifacts whenever possible and ask before expanding the tree. State your confidence level in reviews or experiment notes, flag outdated docs, and raise questions when requirements are ambiguous. Always prefer official MCP documentation sources when you need API or tooling references.
