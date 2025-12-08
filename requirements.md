# HINT Test Suite – Requirements (requirements.md)

This document defines the behavioral and quality requirements for refactoring and extending the HINT project test suite (`src/test/**`).  
All requirements are expressed as user stories with acceptance criteria in EARS-style notation.

---

## 1. Code Style & Documentation Requirements

### R-1 Zero Comment Policy in Test Code

WHEN Python test files under `src/test` are added, modified, or reviewed  
THE SYSTEM SHALL contain no line comments starting with `#` and no inline comments following executable statements.

**Acceptance criteria**

- WHEN a test file is scanned for the pattern `^\s*#.*`  
  THE SYSTEM SHALL report zero matches.

- WHEN a test file is scanned for the pattern `\s+#.*`  
  THE SYSTEM SHALL report zero matches.

- WHEN legacy commented-out test code is encountered  
  THE SYSTEM SHALL remove that code instead of retaining it as comments or disabled blocks in the main branch.

---

### R-2 Docstrings for All Test Functions and Classes

WHEN a test function or test class is defined under `src/test`  
THE SYSTEM SHALL provide a Python docstring in English that follows the agreed template structure.

**Acceptance criteria**

- WHEN a `test_*` function is parsed  
  THE SYSTEM SHALL have a non-empty triple-quoted docstring immediately inside the function body.

- WHEN a test helper class (e.g. `TestRunner`, `Fixtures`) is defined  
  THE SYSTEM SHALL have a non-empty triple-quoted docstring immediately inside the class body.

- WHEN a docstring is inspected  
  THE SYSTEM SHALL include:
  - a short summary line describing what the test verifies,  
  - a detailed description paragraph that specifies component, scenario, and expected outcome,  
  - a “Test Case ID” field,  
  - a “Scenario” field,  
  - `Args` and `Returns` sections, even if arguments are `None`.

---

### R-3 English-only Test Documentation

WHEN test docstrings or string literals describing behavior are authored or updated  
THE SYSTEM SHALL use English language exclusively for these descriptions.

**Acceptance criteria**

- WHEN a docstring is inspected  
  THE SYSTEM SHALL contain only English words and technical identifiers (e.g. class names, config keys).

- WHEN human-readable explanations of scenarios or expectations are added  
  THE SYSTEM SHALL use English, and SHALL NOT mix Korean (or other natural languages) in docstrings or logging messages.

---

### R-4 Replacing Explanatory Comments with Logging

WHEN a complex test scenario requires human-readable explanation  
THE SYSTEM SHALL express that explanation via logging (e.g. `logger.debug`) or docstrings, not via comments.

**Acceptance criteria**

- WHEN a previously commented explanation exists in legacy tests  
  THE SYSTEM SHALL replace it with either:
  - a more descriptive docstring, or  
  - a `logger.debug(...)` / `logger.info(...)` call describing the stage of the test.

- WHEN a reviewer searches for rationale about a non-trivial test setup  
  THE SYSTEM SHALL enable understanding based on docstrings and logs, without relying on comments.

---

## 2. FIRST & Test Behavior Requirements

### R-5 Fast Tests (No Heavy Training in Unit Tests)

WHEN unit tests under `src/test/unit` execute model code  
THE SYSTEM SHALL avoid full backpropagation training loops and SHALL limit work to forward passes or lightweight stubs/mocks.

**Acceptance criteria**

- WHEN a unit test interacts with a trainer or model  
  THE SYSTEM SHALL either:
  - mock the training loop, or  
  - perform a single forward pass with small tensors.

- WHEN CI runs the unit test suite  
  THE SYSTEM SHALL keep total execution time within a reasonable bound (e.g. under a few minutes) without long-running training.

---

### R-6 Isolated Tests (No Real External Side Effects)

WHEN a test touches I/O, network, or external services  
THE SYSTEM SHALL isolate those dependencies by using mocks or temporary test resources.

**Acceptance criteria**

- WHEN a test would otherwise read or write actual data files, network endpoints, or databases  
  THE SYSTEM SHALL instead:
  - mock the relevant functions, or  
  - use temporary directories and files that are cleaned up automatically.

- WHEN CI executes tests multiple times  
  THE SYSTEM SHALL yield identical results without interference from prior runs, thanks to isolation.

---

### R-7 Repeatable Tests (Deterministic Behavior)

WHEN the same test suite is executed multiple times with the same code and configuration  
THE SYSTEM SHALL produce deterministic results and SHALL NOT rely on implicit global state.

**Acceptance criteria**

- WHEN random behavior is used in tests  
  THE SYSTEM SHALL fix random seeds or mock randomness so that outcomes are deterministic.

- WHEN CI re-runs tests on the same commit  
  THE SYSTEM SHALL report the same pass/fail status, assuming no external environment changes.

---

### R-8 Self-validating Tests (No Manual Log Inspection)

WHEN a test verifies CLI tools, training runs, or ETL pipelines  
THE SYSTEM SHALL assert observable outcomes programmatically and SHALL NOT depend on manual log inspection.

**Acceptance criteria**

- WHEN a CLI integration test is executed via `subprocess`  
  THE SYSTEM SHALL capture `stdout` and `stderr` and assert on expected substrings (e.g. `"Usage"`, `"Error"`, `"Hydra"`).

- WHEN training or ETL tests are executed  
  THE SYSTEM SHALL assert on return codes, file outputs, or metric values rather than relying on human review of logs.

---

## 3. Coverage Requirements

### R-9 Statement and Branch Coverage Goal

WHEN the full test suite is run against `src/hint`  
THE SYSTEM SHALL achieve 100% statement coverage and 100% branch coverage for all non-trivial modules within scope of the master test plan.

**Acceptance criteria**

- WHEN a coverage report is generated for `src/hint`  
  THE SYSTEM SHALL show 100% for statements and branches, except for explicitly justified exclusions (e.g. safety guards, platform-specific branches).

- WHEN new modules are added under `src/hint`  
  THE SYSTEM SHALL include them in the coverage target and SHALL add corresponding tests.

---

### R-10 Test Coverage for Telemetry Infrastructure

WHEN the `RichTelemetryObserver` (and related telemetry components) is introduced or modified  
THE SYSTEM SHALL provide dedicated unit tests under `src/test/unit/infrastructure/test_telemetry.py`.

**Acceptance criteria**

- WHEN `RichTelemetryObserver` is initialized in tests  
  THE SYSTEM SHALL verify that a `Console` (or equivalent terminal abstraction) is constructed correctly.

- WHEN `log_metric` or `log_params` are called in tests  
  THE SYSTEM SHALL assert that the underlying Loguru and Rich handlers receive the expected messages or payloads via mocks.

- WHEN Loguru file logging and Rich terminal output are enabled together  
  THE SYSTEM SHALL confirm that no runtime errors or handler conflicts occur during telemetry-related tests.

---

## 4. Edge Case & Exception Handling Requirements

### R-11 Trainer – NaN Loss Handling

WHEN the training service encounters a loss value that becomes `NaN` during a training step  
THE SYSTEM SHALL either raise a clearly documented exception (e.g. `ValueError`) or terminate training gracefully with explicit logging.

**Acceptance criteria**

- WHEN `test_train_handle_nan_loss` is executed  
  THE SYSTEM SHALL simulate a `NaN` loss and assert that:
  - an exception is raised, OR  
  - training stops and a specific log message describes the termination condition.

- WHEN a NaN loss occurs in real training runs  
  THE SYSTEM SHALL not continue silently without any indication in logs or testable outcomes.

---

### R-12 Trainer – Empty DataLoader Handling

WHEN the training service is invoked with an empty `DataLoader`  
THE SYSTEM SHALL handle the situation predictably and SHALL NOT crash with obscure errors.

**Acceptance criteria**

- WHEN `test_train_empty_dataloader` is executed  
  THE SYSTEM SHALL assert that:
  - either a well-defined exception (e.g. `ValueError` or `ConfigurationError`) is raised, OR  
  - the trainer logs a clear message and exits gracefully.

- WHEN users misconfigure the dataset size to zero  
  THE SYSTEM SHALL expose this failure mode clearly through test-verified behavior.

---

### R-13 ETL – Missing Required Columns

WHEN ETL components receive a Parquet (or table) input missing required columns (e.g. `SUBJECT_ID`, `ICUSTAY_ID`, or other mandatory fields)  
THE SYSTEM SHALL raise a domain-specific validation error rather than failing with generic low-level exceptions.

**Acceptance criteria**

- WHEN `test_missing_required_columns` is executed on ETL modules (e.g. `test_assembler.py`, `test_ventilation.py`)  
  THE SYSTEM SHALL assert that a `DataValidationError` (or equivalent explicit error) is raised.

- WHEN upstream changes remove or rename critical columns  
  THE SYSTEM SHALL reveal the problem via failing validation tests with clear error types and messages.

---

### R-14 Config – Invalid Paths Handling

WHEN the configuration specifies non-existent data or artifact paths  
THE SYSTEM SHALL detect the problem early and report a clear configuration-related error.

**Acceptance criteria**

- WHEN `test_invalid_config_paths` is executed  
  THE SYSTEM SHALL assert that:
  - a `FileNotFoundError` or `ConfigurationError` is raised, and  
  - the error message contains the missing path.

- WHEN users run the system with wrong paths  
  THE SYSTEM SHALL fail fast with helpful diagnostics, as guaranteed by tests.

---

### R-15 Infrastructure – Corrupted HDF5 File Handling

WHEN the infrastructure layer attempts to load a corrupted HDF5 file  
THE SYSTEM SHALL fail with a controlled, explicit error instead of an unhandled low-level exception.

**Acceptance criteria**

- WHEN `test_hdf5_corrupted_file` is executed  
  THE SYSTEM SHALL provide a reproducible simulated corrupted HDF5 file and assert that:
  - a specific exception type is raised, OR  
  - error handling logic logs a clear message and aborts the operation.

- WHEN real corrupted HDF5 files appear in production-like runs  
  THE SYSTEM SHALL respond in the same controlled way as defined and validated by the test.

---

## 5. Logging & Observability Requirements

### R-16 Coordinated Loguru and Rich Setup in Test Runner

WHEN the test runner initializes logging (e.g. in `src/test/runner.py`)  
THE SYSTEM SHALL configure Loguru and Rich in a way that prevents handler conflicts and ensures clear separation of file vs terminal output.

**Acceptance criteria**

- WHEN `_setup_logging` is invoked in the runner  
  THE SYSTEM SHALL call `logger.remove()` to clear default Loguru handlers before adding custom ones.

- WHEN terminal logging is configured  
  THE SYSTEM SHALL use a Rich-compatible sink (e.g. `Console` or `stderr`) with a simple format that does not conflict with Rich progress bars.

- WHEN file logging is configured  
  THE SYSTEM SHALL direct Loguru logs into the Hydra-managed `run_dir` (or equivalent), rather than creating ad-hoc log file locations.

---

### R-17 Logging as Stage Indicator in Tests

WHEN a test progresses through multiple logical stages (e.g. setup, execution, assertion)  
THE SYSTEM SHALL log stage information with sufficient granularity to reconstruct the step-by-step progression.

**Acceptance criteria**

- WHEN a complex integration or E2E test runs  
  THE SYSTEM SHALL emit log entries that indicate key stages such as “setup test data”, “invoke trainer”, “verify outputs”.

- WHEN a failure occurs in a multi-stage test  
  THE SYSTEM SHALL allow developers to infer the failing stage from logged messages without needing comments.

---

## 6. Configuration & Infrastructure Requirements

### R-18 Pathlib Usage in Test and Runner Code

WHEN file system paths are used in test code or test runner utilities  
THE SYSTEM SHALL use `pathlib.Path` and path operations instead of `os.path` utilities.

**Acceptance criteria**

- WHEN new test utility code manipulates paths  
  THE SYSTEM SHALL use expressions such as `Path("data") / "processed"`.

- WHEN directories are created in tests  
  THE SYSTEM SHALL use `Path.mkdir` with appropriate flags instead of `os.makedirs`.

- WHEN files are opened  
  THE SYSTEM SHALL use `Path.open()` instead of `open()` with string paths where practical.

---

### R-19 Hydra-based Test Runner Configuration

WHEN the test runner is executed  
THE SYSTEM SHALL obtain configuration via a Hydra entrypoint and a `test_config.yaml` (or equivalent), not via `argparse`.

**Acceptance criteria**

- WHEN the runner’s main entry function is inspected  
  THE SYSTEM SHALL be decorated with `@hydra.main(...)` and SHALL accept a Hydra config object rather than CLI-parsed arguments.

- WHEN test configuration is changed (e.g. paths, log level, dataset size)  
  THE SYSTEM SHALL apply those changes through the Hydra config hierarchy and test them via dedicated runner tests.

---

### R-20 No `argparse` in Test Runner

WHEN command-line arguments are needed for the test runner  
THE SYSTEM SHALL express them as Hydra configuration options instead of using `argparse`.

**Acceptance criteria**

- WHEN `src/test/runner.py` is searched for `argparse` usage  
  THE SYSTEM SHALL show no imports or references to `argparse`.

- WHEN environment-specific options (e.g. selecting subsets of tests, toggling integration tests) are required  
  THE SYSTEM SHALL expose them via Hydra config parameters and SHALL be covered by tests that validate configuration-driven behavior.

---

## 7. Test Suite Organization & Phase Requirements

### R-21 Layer-aware Test Organization

WHEN tests are added or moved  
THE SYSTEM SHALL organize them into clear directories that mirror the DDD layers (Foundation, Domain, Infrastructure, Services, Integration/E2E).

**Acceptance criteria**

- WHEN the `src/test` tree is inspected  
  THE SYSTEM SHALL provide a discernible mapping from each test module to its corresponding layer and production module.

- WHEN a new production module is introduced in a specific layer  
  THE SYSTEM SHALL add or extend tests in the matching test subdirectory.

---

### R-22 Phase-wise Refactoring Compliance

WHEN refactoring work proceeds through the defined phases (Infra → Unit Core → Unit Service → Integration/E2E)  
THE SYSTEM SHALL ensure that each phase leaves the test suite in a passing, coherent, and style-compliant state.

**Acceptance criteria**

- WHEN Phase 1 (Infra) changes are completed  
  THE SYSTEM SHALL have:
  - zero comments,  
  - standardized docstrings and logging in runner/util tests,  
  - passing CI for those tests.

- WHEN Phases 2–4 are completed sequentially  
  THE SYSTEM SHALL maintain green CI status and SHALL not reintroduce style or coverage regressions already addressed in earlier phases.

---
