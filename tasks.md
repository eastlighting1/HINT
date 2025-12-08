# HINT Test Suite – Implementation Plan (tasks.md)

This document breaks down the refactor of the HINT test suite into concrete, trackable tasks.  
Tasks are grouped by phase and linked back to `requirements.md` and `design.md`.

Legend:

- [ ] Not started  
- [~] In progress  
- [x] Done  

---

## Phase 0 – Baseline & Tooling

### [ ] T0.1 – Establish Baseline Coverage and Test Status

- **Description**  
  Run the existing test suite and coverage to capture the current baseline for `src/hint` and `src/test`.

- **Steps**
  - Run `pytest` on the current project without modifications.
  - Run coverage (e.g. `pytest --cov=src/hint --cov-report=term-missing`).
  - Save coverage report (HTML/term output) to `docs/` or `artifacts/` directory.
  - Document the current statement/branch coverage in a short markdown note.

- **Expected Outcome**
  - A documented baseline of coverage and failing tests (if any) for comparison after refactors.

- **Dependencies**
  - None.

- **Related Requirements**
  - R-9 (coverage goal, context for improvement).

---

### [ ] T0.2 – Introduce Central Test Make/Script Entry

- **Description**  
  Provide an easy developer entrypoint (e.g. `make test` or `poe test`) that will later call the Hydra-based runner.

- **Steps**
  - Add a minimal script or Makefile target that currently calls `pytest` directly.
  - Ensure CI uses this entrypoint (or document how it will be switched to the new runner).

- **Expected Outcome**
  - Single canonical command to run tests locally and in CI, which can be transparently swapped to use the new runner in Phase 1.

- **Dependencies**
  - T0.1 (optional but recommended).

- **Related Requirements**
  - Supports R-21, R-22 (organized, phase-wise evolution).

---

## Phase 1 – Infra: Runner, Logging, Pathlib, Global Style

### [ ] T1.1 – Implement Hydra-based Test Runner (`src/test/runner.py`)

- **Description**  
  Refactor `runner.py` into a Hydra-driven entrypoint that orchestrates pytest execution based on `configs/test_config.yaml`.

- **Steps**
  - Decorate the main entry function in `runner.py` with `@hydra.main(config_path="../configs", config_name="test_config")` (or equivalent relative path).
  - Accept a Hydra config object (e.g. `cfg`) instead of parsing CLI arguments.
  - Build pytest argument list based on `cfg` fields (e.g. which test groups to run, coverage targets).
  - Invoke pytest programmatically (e.g. `pytest.main(pytest_args)`).
  - Return pytest exit code as the runner’s exit code.

- **Expected Outcome**
  - `python -m src.test.runner` runs tests according to `test_config.yaml` without using `argparse`.

- **Dependencies**
  - T0.2 (entrypoint alignment).

- **Related Requirements**
  - R-18, R-19, R-20 (Hydra-based runner, no argparse, Pathlib usage in runner).

---

### [ ] T1.2 – Define `configs/test_config.yaml` for Test Control

- **Description**  
  Add and wire up `configs/test_config.yaml` as the source of truth for test selection, logging config, and coverage settings.

- **Steps**
  - Add `test_config.yaml` with sections such as:
    - `tests: { run_unit: true, run_integration: true, run_e2e: true }`
    - `paths: { test_root: "src/test", src_root: "src/hint" }`
    - `logging: { level: "INFO" }`
    - `coverage: { targets: ["src/hint"], omit: [] }`
  - Update `runner.py` to read from these config fields.
  - Document how to override these options via Hydra CLI (e.g. `python -m src.test.runner tests.run_e2e=false`).

- **Expected Outcome**
  - Test behavior (which suites run, log level, coverage) can be controlled without code changes.

- **Dependencies**
  - T1.1.

- **Related Requirements**
  - R-19, R-21, R-22.

---

### [ ] T1.3 – Implement Coordinated Loguru + Rich Setup in Runner

- **Description**  
  Configure logging in `_setup_logging(cfg)` to avoid conflicts and satisfy observability requirements for tests.

- **Steps**
  - In `runner.py`, implement `_setup_logging(cfg)`:
    - Call `logger.remove()` to clear default Loguru handlers.
    - Create a Hydra `run.dir`-based file path via `Path` for the log file (e.g. `run_dir / "tests.log"`).
    - Add Loguru file sink pointing to the log file with the requested format.
    - Initialize a Rich `Console` for terminal output.
    - Add a Loguru sink that writes to the console via Rich, with simple formatting compatible with progress bars.
  - Ensure `_setup_logging` is called before invoking pytest.
  - Add minimal unit tests for `_setup_logging` behavior, where possible, using mocks.

- **Expected Outcome**
  - Tests have unified, non-conflicting logging: structured log file via Loguru + nicely formatted Rich output in the terminal.

- **Dependencies**
  - T1.1, T1.2.

- **Related Requirements**
  - R-16, R-17.

---

### [ ] T1.4 – Enforce Pathlib Usage in Test Infrastructure

- **Description**  
  Replace `os.path` and bare `open()` usages in test infrastructure (`runner.py`, `conftest.py`, `utils`) with `pathlib.Path`.

- **Steps**
  - Search `src/test` for `os.path`, `os.makedirs`, and bare `open(` with string paths.
  - Refactor to use `Path` and methods like `/`, `mkdir`, `open`.
  - Ensure cross-platform correctness (e.g. not relying on OS-specific separators).
  - Run tests to verify no regressions.

- **Expected Outcome**
  - Test-related file system operations use `pathlib.Path` exclusively.

- **Dependencies**
  - T1.1 (runner in particular).

- **Related Requirements**
  - R-18.

---

### [ ] T1.5 – Implement Global Zero-comment & Docstring Style Checks

- **Description**  
  Introduce automated style checks that enforce the zero-comment policy and docstring presence for tests.

- **Steps**
  - Add a dedicated test file (e.g. `src/test/unit/test_style_tests.py`) or a small utility:
    - Walk all `.py` files under `src/test`.
    - Use regex to assert absence of:
      - Full-line comments (`^\s*#.*`).
      - Inline comments (`\s+#.*`).
  - Add a second style check that:
    - For each `test_*.py` file, inspects AST for functions starting with `test_` and classes starting with `Test`.
    - Assert that each of these objects has a docstring.
  - Integrate these tests into the test suite so any violation fails CI.

- **Expected Outcome**
  - Automated enforcement of zero-comment policy and mandatory docstrings across `src/test`.

- **Dependencies**
  - None (can be developed in parallel with other Phase 1 tasks, but executed after initial docstring migrations).

- **Related Requirements**
  - R-1, R-2, R-3, R-4.

---

### [ ] T1.6 – Refactor Global `conftest.py` and `utils` to New Style

- **Description**  
  Make `src/test/conftest.py` and `src/test/utils/*` compliant with: zero comments, English docstrings, Pathlib, and logging.

- **Steps**
  - Remove all comments from these modules.
  - Add or standardize docstrings using the test docstring template for any helper functions/classes.
  - Replace `os` path handling with `Path`.
  - Ensure logging uses Loguru/Rich as designed (if they log anything).
  - Run style tests (T1.5) and fix violations.

- **Expected Outcome**
  - Shared fixtures and utilities fully comply with the global test style and infrastructure constraints.

- **Dependencies**
  - T1.3, T1.4, T1.5.

- **Related Requirements**
  - R-1, R-2, R-3, R-18, R-17.

---

## Phase 2 – Unit Core: Foundation, Domain, Infrastructure

### [ ] T2.1 – Foundation Tests Refactor (Configs, DTOs, Exceptions)

- **Description**  
  Align all foundation unit tests with new style and ensure full coverage of `foundation/configs.py`, `dtos.py`, `exceptions.py`, `interfaces.py` where applicable.

- **Steps**
  - Update `src/test/unit/foundation/test_configs.py`, `test_dtos.py`, `test_exceptions.py`:
    - Remove comments.
    - Add standardized docstrings to every test function and test class.
    - Add missing edge-case tests (invalid configs, DTO invalid inputs, exception behaviors).
  - Confirm each foundation module reaches 100% statement and branch coverage.
  - Update or add tests for interfaces if they have behavior worth verifying.

- **Expected Outcome**
  - Foundation layer tests are FIRST-compliant, self-validating, and cover all branches of foundation modules.

- **Dependencies**
  - T1.5, T1.6.

- **Related Requirements**
  - R-5, R-6, R-7, R-9, R-11–R-15 as relevant for foundation-level validation.

---

### [ ] T2.2 – Domain Tests Refactor (Entities, Value Objects)

- **Description**  
  Ensure domain tests for `entities.py` and `vo.py` are pure, deterministic, and fully covered with new docstring style.

- **Steps**
  - Update `src/test/unit/domain/test_entities.py` and `test_vo.py`:
    - Remove any comments, enforce English docstrings.
    - Add tests for all invariants and error conditions in domain objects.
    - Confirm that tests do not touch I/O, configs, or external frameworks.
  - Verify 100% statement/branch coverage for the domain modules.

- **Expected Outcome**
  - Domain layer is fully specified and enforced by pure, fast unit tests.

- **Dependencies**
  - T1.5, T1.6.

- **Related Requirements**
  - R-5, R-6, R-7, R-9.

---

### [ ] T2.3 – Infrastructure Tests: Components, Registry, Telemetry, Networks

- **Description**  
  Refactor and complete unit tests for infrastructure modules, ensuring proper handling of edge cases and telemetry integration.

- **Steps**
  - Update `src/test/unit/infrastructure/test_components.py`, `test_registry.py`, `test_networks.py`, `test_datasource.py`, `test_telemetry.py`:
    - Enforce zero comments and docstring style.
    - For `test_networks.py`:
      - Ensure only small synthetic tensors and forward passes are tested (no training loops).
      - Add tests for network misconfigurations (e.g., missing categorical vocab sizes) to cover failure branches.
    - For `test_datasource.py`:
      - Use temporary directories and synthetic data; no real dataset paths.
      - Add tests for HDF5 and other storage error paths, deferring the corrupted-file scenario to integration tests (T3.3) if needed.
    - For `test_telemetry.py`:
      - Mock `Console` and Loguru sinks to verify `log_metric` and `log_params` calls.
  - Confirm 100% coverage per infrastructure module.

- **Expected Outcome**
  - Infrastructure layer behavior and failure modes are rigorously unit-tested and aligned with telemetry/logging design.

- **Dependencies**
  - T1.3, T1.4, T1.5.

- **Related Requirements**
  - R-5, R-6, R-7, R-9, R-10, R-15, R-16, R-17.

---

### [ ] T2.4 – App Factory Tests (`test_factory.py`)

- **Description**  
  Ensure `src/test/unit/app/test_factory.py` fully validates application wiring logic.

- **Steps**
  - Enforce style (no comments, English docstrings).
  - Add tests that:
    - Construct the app components via `factory.py` using a Hydra-configured or stub config.
    - Assert that correct instances are created for ETL, ICD, training services, and infrastructure dependencies.
  - Confirm full coverage for `app/factory.py`.

- **Expected Outcome**
  - The app wiring is stable and safeguarded by fast, deterministic unit tests.

- **Dependencies**
  - T1.2 (Hydra config), T2.1–T2.3 (underlying layers stable).

- **Related Requirements**
  - R-5, R-6, R-7, R-9, R-21.

---

## Phase 3 – Unit Services: ETL, ICD, Training

### [ ] T3.1 – ETL Service Components: Edge Cases & Validation

- **Description**  
  Expand ETL component tests to cover missing columns, invalid inputs, and configuration-related failure modes.

- **Steps**
  - Update `src/test/unit/services/etl/components/test_*.py`:
    - Remove comments and ensure docstrings.
    - For each component (`assembler`, `labels`, `notes`, `outcomes`, `static`, `tensor`, `timeseries`, `ventilation`):
      - Add tests for:
        - Required-column presence; missing columns should raise `DataValidationError` (or equivalent).
        - Invalid value types or inconsistent lengths.
    - Implement `test_missing_required_columns` in appropriate modules to cover R-13 explicitly.
  - Use synthetic tables built via `synthetic_data` utilities to avoid real MIMIC data.

- **Expected Outcome**
  - ETL components fail fast and loudly for schema issues, with behavior fully validated by unit tests.

- **Dependencies**
  - T2.1, T2.3.

- **Related Requirements**
  - R-11–R-15 (especially R-13, R-14), R-5, R-6, R-7.

---

### [ ] T3.2 – ICD Service Tests (`services/icd/service.py`)

- **Description**  
  Ensure ICD service tests cover both typical and error scenarios while remaining fast via mocking of heavy models.

- **Steps**
  - Refactor `src/test/unit/services/icd/test_service.py`:
    - Enforce style and docstrings.
    - Mock the underlying MedBERT encoder or use lightweight stubs.
    - Add tests for:
      - Successful ICD predictions given valid inputs.
      - Handling of empty or malformed text inputs.
      - Configuration errors (e.g., missing model name or paths).

- **Expected Outcome**
  - ICD service is well specified at the service boundary and is unit-tested independently of actual large models.

- **Dependencies**
  - T2.3 (infrastructure: networks/datasource/telemetry).

- **Related Requirements**
  - R-5, R-6, R-7, R-9.

---

### [ ] T3.3 – Training Service Tests: NaN Loss, Empty DataLoader, Edge Paths

- **Description**  
  Cover critical edge-case behavior in training: NaN loss, empty dataloader, and other misconfigurations.

- **Steps**
  - Update `src/test/unit/services/training/test_trainer.py` and `test_evaluator.py`:
    - Enforce style and docstrings using the standardized template.
    - Implement:
      - `test_train_handle_nan_loss`: simulate NaN loss (e.g., via mock model or loss function) and assert:
        - A documented exception (e.g., `ValueError`) is raised, or
        - Training stops with a clear log message and exit condition.
      - `test_train_empty_dataloader`: create a DataLoader with zero batches and assert:
        - A clear exception (`ValueError`/`ConfigurationError`) is raised, or
        - Training exits gracefully with logs.
      - Additional tests for misconfigured optimizer or scheduler where relevant.
    - Ensure that tests do not perform full training; they should use minimal forward passes or mocks.
  - Confirm 100% coverage for `trainer.py` and `evaluator.py`.

- **Expected Outcome**
  - Training services handle pathological cases predictably and transparently, with behavior fully specified and enforced by tests.

- **Dependencies**
  - T2.3, T2.4.

- **Related Requirements**
  - R-5, R-6, R-7, R-8, R-11, R-12.

---

## Phase 4 – Integration & End-to-end

### [ ] T4.1 – Integration Tests for Infrastructure (HDF5, Persistence)

- **Description**  
  Strengthen integration tests around HDF5 datasource and model persistence to include corrupted-file handling and recovery behavior.

- **Steps**
  - Update `src/test/integration/infrastructure/test_hdf5_datasource.py`:
    - Add scenario generating a corrupted HDF5 file via `synthetic_data` utilities.
    - Assert that attempting to read this file raises a domain-specific error or a well-documented exception.
  - Update `test_model_persistence.py`:
    - Ensure save-load cycles are tested for correctness.
    - Include negative tests (e.g., missing file, wrong path).
  - Use temporary directories and Pathlib.

- **Expected Outcome**
  - Robust integration coverage of HDF5 and persistence behavior, including error handling for corrupted or missing files.

- **Dependencies**
  - T2.3, T3.1.

- **Related Requirements**
  - R-6, R-7, R-9, R-15.

---

### [ ] T4.2 – Integration Tests for Workflows (ETL, Full Training Loop)

- **Description**  
  Ensure ETL and training workflow integration tests reflect realistic mini-pipelines and are self-validating.

- **Steps**
  - Update `src/test/integration/workflows/test_etl_execution.py`:
    - Use synthetic source tables for ETL.
    - Assert that expected intermediate outputs (e.g. processed Parquet/HDF5 files) are created in temporary directories.
    - Assert that logs contain stage markers (e.g. “start ETL”, “ETL completed”).
  - Update `test_full_training_loop.py`:
    - Use small synthetic dataset and simple network configuration.
    - Assert:
      - Training completes without error.
      - At least a minimal metric is computed and logged.
    - Ensure limited number of epochs/steps for speed.
  - Ensure tests read `test_config.yaml` where paths or dataset sizes are controlled.

- **Expected Outcome**
  - Representative integration tests validating end-to-end ETL and training workflows on downsized data, without manual inspection.

- **Dependencies**
  - T3.1, T3.3, T1.1–T1.3.

- **Related Requirements**
  - R-5, R-6, R-7, R-8, R-9, R-17, R-21.

---

### [ ] T4.3 – E2E CLI Tests (`test_cli_entrypoint.py`)

- **Description**  
  Harden CLI end-to-end tests to verify black-box behavior, including help output, error handling, and pipeline execution.

- **Steps**
  - Update `src/test/e2e/test_cli_entrypoint.py`:
    - Use `subprocess` to invoke the CLI via `python -m hint.app.main` or equivalent entrypoint.
    - Add tests for:
      - `--help` output: assert presence of usage text and key options.
      - Invalid arguments or missing configs: assert non-zero exit codes and error messages.
      - A minimal full pipeline run using synthetic data and `test_config.yaml` overrides.
    - Capture `stdout` and `stderr` and assert on key substrings (e.g. “Hydra”, “Error”, “Completed”).
  - Ensure E2E tests remain fast by using tiny datasets and a minimal number of epochs.

- **Expected Outcome**
  - E2E tests serve as a black-box verification that the CLI is wired correctly and behaves predictably in success and failure modes.

- **Dependencies**
  - T1.1–T1.3, T4.2.

- **Related Requirements**
  - R-8, R-16, R-17, R-19, R-22.

---

## Phase 5 – Finalization & Quality Gates

### [ ] T5.1 – Verify 100% Coverage & Tighten Coverage Thresholds

- **Description**  
  Confirm that statement and branch coverage for `src/hint` is at 100% and enforce this as a CI gate.

- **Steps**
  - Run coverage again after all refactors: `pytest --cov=src/hint --cov-branch`.
  - Inspect report and fix any uncovered branches or lines with targeted tests.
  - Configure coverage thresholds in pyproject/CI (e.g. `fail_under = 100` for statements and branches).
  - Document the final coverage metrics.

- **Expected Outcome**
  - CI fails when coverage falls below 100% for `src/hint`, enforcing the coverage requirement.

- **Dependencies**
  - All Phase 1–4 tasks.

- **Related Requirements**
  - R-9.

---

### [ ] T5.2 – Ensure FIRST Compliance Across Entire Test Suite

- **Description**  
  Perform a holistic pass to ensure the test suite adheres to FIRST principles: Fast, Isolated, Repeatable, Self-validating, Thorough.

- **Steps**
  - Review long-running tests and:
    - Reduce epochs/steps.
    - Mock heavy dependencies where possible.
  - Confirm no tests rely on external state (network, real files) without isolation.
  - Confirm tests do not require manual log inspection.
  - Ensure edge-case scenarios (error paths, invalid configurations) are covered for all critical modules.

- **Expected Outcome**
  - The test suite is performant, robust, and reliable for repeated CI executions.

- **Dependencies**
  - T3.1–T3.3, T4.1–T4.3.

- **Related Requirements**
  - R-5, R-6, R-7, R-8, R-11–R-15, R-21, R-22.

---

### [ ] T5.3 – Documentation Update (README, Dev Notes for Tests)

- **Description**  
  Document how to run, extend, and reason about the HINT test suite based on the new design.

- **Steps**
  - Update `README.md` (or add `docs/testing.md`) to include:
    - How to run tests (`python -m src.test.runner`, `make test`, etc.).
    - How Hydra `test_config.yaml` is structured and can be overridden.
    - Rules for writing new tests (zero comments, docstrings, Pathlib, logging patterns).
  - Add a short section describing the testing architecture (unit vs integration vs e2e) and DDD mapping.
  - Optionally embed or reference `requirements.md` and `design.md`.

- **Expected Outcome**
  - New contributors can quickly understand and conform to the testing architecture and rules.

- **Dependencies**
  - Completion of at least Phases 1–4.

- **Related Requirements**
  - All, as this provides the human-facing guide to the system already encoded by tests.

---
