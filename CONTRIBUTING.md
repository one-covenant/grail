## Contributing Guide

Thanks for helping make this project better. Please keep changes focused, well-tested, and small.

### Top ways to contribute
- **Hardware support and inference efficiency**: Before starting work on a new backend or
  hardware target, please open an issue or discussion to align on need and design. We have not
  finalized the multi-backend architecture yet. Add or improve support for your
  accelerator/GPU/CPU. Optimize kernels, memory layout, quantization, and runtime selection.
  Include benchmarks, configuration notes, and a safe CPU/GPU fallback path.
- **High-impact bug fixes**: Pick issues labeled `bug` or `critical`. Provide a minimal
  repro and a regression test.
- **Docs and tests**: Clarify tricky areas and strengthen test coverage.

### Current priorities
- Inference verification bug fixes
- General bug fixes

### Getting started
- **Environment**: Python 3.9–3.11. We use `uv` for dependency management.
  - Install deps: `uv sync --all-extras`
  - Add deps: `uv add package==X.Y.Z` (pin exact versions) and then `uv lock`
  - Run tasks: `uv run <command>` (e.g., `uv run pytest -q`)
- **Before large changes**: Open an issue to discuss scope and design.
- **Branches**:
  - `feat/hw-<target>`: hardware backends or runtime integrations
  - `feat/<area>-<short>`: non-hardware features
  - `fix/<area>-<short>`: bug fixes
  - `chore/<area>-<short>`: maintenance and tooling
  - `docs/<area>-<short>`: documentation-only changes
  - Examples:
    - `feat/hw-rocm`
    - `feat/scheduler-batch-padding`
    - `fix/inference-verification-mismatch`
    - `chore/ci-uv-sync`
    - `docs/hardware-setup-mlx`

### Commit and PR conventions
- Use **Conventional Commits** for commit messages and PR titles (e.g., `feat: ...`, `fix: ...`,
  `perf: ...`, `docs: ...`; use `!` or `BREAKING CHANGE:` for breaking changes). See
  [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/).

### Pull request checklist
- **Tests**: Add/extend tests; include a small benchmark or profile for inference/hardware changes.
- **Docs**: Update README/docs where behavior or usage changes.
- **Deps**: Reuse existing libraries when possible; if adding, pin exact versions and run `uv lock`.
- **Local checks**: Formatting, linting, type checks, and tests must pass locally.
  - Format: `uv run black .`
  - Lint: `uv run flake8`
  - Types: `uv run mypy`
  - Tests: `uv run pytest -q`
  - Note: CI for tests and type checks is **planned** but not yet enabled; please run locally.

### Style guide (Python)
- **PEP 8, 100 chars**: Keep lines ≤100 characters.
- **Type hints**: Add annotations for all public functions (arguments and return values).
- **Imports**: Group as stdlib, third-party, local. No wildcard imports. Use relative imports within
  the `grail` package.
- **Logging**: Use `logging.getLogger(__name__)`. No prints. Include context in messages.
- **Comments and docstrings**: Add clear docstrings and explanatory comments for non-obvious or
  complex code paths. Keep simple code uncommented but name things clearly.
- **Concurrency/IO**: Prefer `asyncio` for IO, avoid blocking the event loop, and set timeouts for
  network calls.
- **Security**: Never log secrets. Load configuration from environment (e.g., `python-dotenv`).
- **ML conventions**: Use `transformers.AutoTokenizer` / `AutoModelForCausalLM`. Store weights with
  `safetensors`. Avoid `pickle` for untrusted data.

### Hardware backend contributions
Note: The multi-backend/hardware architecture is under design. Please begin by opening an
issue or discussion to propose the use case and desired scope so we can align on interfaces
and testing strategy before implementation.
- **Deliverables**:
  - Device detection/configuration and a pluggable inference path for your hardware.
  - Benchmarks (model, shapes, batch size, dtype) and expected vs. achieved numbers.
  - Fallback path and graceful degradation when unsupported features are encountered.
  - Tests covering correctness and a short doc on setup/usage.

### Bug reports
- Provide a minimal repro script, expected vs actual behavior, environment info (Python, CUDA/driver,
  library versions), and relevant logs.

We appreciate every contribution—thank you for helping improve performance, stability, and reach.



