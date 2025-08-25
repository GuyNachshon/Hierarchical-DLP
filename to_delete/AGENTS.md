# Repository Guidelines

## Project Structure & Module Organization
- Code: `HRM/hrm_dlp/` (core logic), `HRM/utils/` (helpers), `HRM/models/` (schemas).
- Scripts: `HRM/scripts/` (e.g., `agentic_data_generator.py`, `task_dashboard.py`).
- Config: `HRM/config/` (architecture/run-time). Example env in `.env`.
- Data & assets: `HRM/data/`, `HRM/assets/`, sample data in root `data/`.
- Docs: `README_DLP.md`, `HRM DLP Project Overview.md`, `CLAUDE.md`, `GEMINI.md`.
- Tests: co-located under `HRM/` and `HRM/scripts/` as `test_*.py`.

## Build, Test, and Development Commands
- Create env: `uv venv && source .venv/bin/activate` (or `python -m venv .venv`).
- Install deps: `uv sync` (preferred) or `pip install -e .` or `pip install -r HRM/requirements.txt`.
- Quick start: `python HRM/quick_start_dlp.py` (runs a minimal demo).
- Generator help: `python HRM/scripts/agentic_data_generator.py --help`.
- Example tests: `python HRM/test_fixed_agentic.py`, `python HRM/scripts/test_batch_quick.py`.

## Coding Style & Naming Conventions
- PEP 8, 4-space indentation; add type hints where practical.
- Names: functions/modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Formatting: Black (line length 88) and Ruff if available.
- Prefer `logging` over `print` in library code.

## Testing Guidelines
- Keep tests deterministic and offline; mock LLM/network calls.
- Name tests `test_*.py`; place near the code (under `HRM/` or `HRM/scripts/`).
- Run directly: `python path/to/test_file.py` or with `pytest` if configured.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits, e.g., `feat(scripts): add batch ETA fix`, `fix(hrm_dlp): robust span extraction`.
- PRs: clear description, linked issues, repro steps, sample logs/output, and any config/data path changes; include dashboard screenshots when relevant.
- Keep changes minimal and focused; update docs/configs when behavior changes.

## Security & Configuration Tips
- Store secrets in `.env` (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`). Do not commit keys.
- Write datasets/artifacts under `HRM/data/` (or set `AgenticConfig(output_dir=...)`). Review generated data before sharing.

## Architecture & Extensibility
- Pipeline: Manager Agent → Specialized Agents (legal/finance/hr/security/casual) → Conversational Agents; orchestration via BatchTracker/Monitor and `task_dashboard`.
- Extend by adding/registering agents in `HRM/scripts/agentic_data_generator.py`, updating span rules in `HRM/scripts/semantic_obfuscation.py`, and extending labeling in `_generate_document_labels`.
# Repository Guidelines

## Project Structure & Module Organization
- Core code: `HRM/hrm_dlp/` (pipeline logic), `HRM/utils/` (helpers), `HRM/models/` (schemas).
- Agentic generator: `HRM/scripts/agentic_data_generator/` (agents, batch, coordinator, config).
- Scripts: `HRM/scripts/` (e.g., `validate_structured_output.py`, small utilities).
- Config: `.env` for API keys; optional `HRM/config/` defaults.
- Data & runs: `data/runs/<session>/` (batch_inputs/outputs, checkpoints); final files in `data/dlp_*`.
- Docs: `README_DLP.md`, project overviews, model notes. Tests live under `HRM/` and `HRM/scripts/` as `test_*.py`.

## Build, Test, and Development Commands
- Create env: `uv venv && source .venv/bin/activate` (or `python -m venv .venv`).
- Install deps: `uv sync` (preferred) or `pip install -r HRM/requirements.txt`.
- Quick start: `python HRM/quick_start_dlp.py`.
- Generator (regular mode): `python HRM/scripts/agentic_data_generator/main.py`.
  - Flags: `--no-auto-retrieve` (submit-only), `--sequential-splits`, `--disable-batch`.
- Validate OpenAI schema: `python HRM/scripts/validate_structured_output.py --model gpt-4o`.
- Example tests: `python HRM/scripts/agentic_data_generator/test_fixed_system.py`.

## Coding Style & Naming Conventions
- PEP 8; 4‑space indentation; add type hints where helpful.
- Names: functions/modules `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Formatting: Black (line length 88) and Ruff if available.
- Prefer `logging` over `print` in library code.

## Testing Guidelines
- Keep tests deterministic and offline; mock LLM/network I/O.
- Name tests `test_*.py`; co‑locate near the code under `HRM/` or `HRM/scripts/`.
- Run via `python path/to/test_file.py` (or `pytest` if configured).

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat(batch): add OpenAI polling`, `fix(coordinator): preserve session files`).
- PRs: clear description, linked issues, repro steps, sample logs/output, and any config/data path changes; include dashboard or run screenshots when relevant.
- Keep diffs minimal and focused; update docs/config when behavior changes.

## Security & Configuration Tips
- Put secrets in `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.); never commit keys.
- Write datasets/artifacts under `data/runs/` or `HRM/data/` (or set `AgenticConfig(output_dir=...)`). Review generated content before sharing.

## Architecture Overview
- Flow: ManagerAgent → Domain Agents → (optional) Conversational/Augmentation → Batch processing.
- Batch: one batch per split, consistent model; pause‑aware polling and recovery. Inputs/outputs under `data/runs/<session>/`.
