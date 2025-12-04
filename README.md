# HINT: Hierarchical ICD-aware Network for Time-series Intervention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset: MIMIC-III](https://img.shields.io/badge/Dataset-MIMIC--III-blue.svg)](https://physionet.org/content/mimiciii/)

> **Author:** Donghyeon Kim  
> **Institution:** Gachon University, School of Computing

---

## 📖 Overview

**HINT** is a hierarchical Clinical Decision Support System (CDSS) designed to predict mechanical ventilation interventions in the Intensive Care Unit (ICU).

Unlike traditional models that rely solely on vital signs, HINT integrates high-level diagnostic context with low-level time-series data. It addresses the challenges of **sparse ICU data**, **incomplete diagnostic labels (partial labels)**, and **extreme class imbalance** in intervention events.

### Key Features

- **Hierarchical Structure:** Two-stage pipeline consisting of *Automated ICD Coding* and *Intervention Prediction*.
- **Partial Label Learning:** Handles incomplete and noisy EMR diagnostic records effectively.
- **Context-Adaptive:** Uses an **ICD-Conditioned Gating Mechanism** to dynamically reweight physiological features based on the patient's diagnostic context.
- **Imbalance-Robust:** Achieves strong performance on rare events (ONSET/WEAN) using class-balanced focal loss and group-wise Temporal Convolutional Networks (TCNs).
- **Explainable (XAI):** Provides global (SHAP) and local (LIME) explanations to support clinical decision-making.

---

## 🏗️ Architecture

The system operates in two main phases:

1. **Automated ICD Coding Module**
   - **Inputs:** Candidate ICD-9 code sets and static/numeric features.
   - **Method:** Partial label learning with BERT-based text encoders.
   - **Output:** Inferred admission-level representative ICD context.

2. **Intervention Prediction Module**
   - **Inputs:** 3-channel time-series tensor (**VAL**, **MSK**, **DELTA**) and inferred ICD context.
   - **Method:** **Group-wise TCN** (Temporal Convolutional Network) with an **ICD-Conditioned Gating Mechanism**.
   - **Output:** Probability of 4 states: `ONSET`, `WEAN`, `STAY ON`, `STAY OFF`.

<div align="center">
  <img src="assets/pipeline_diagram.png" alt="HINT Architecture" width="800"/>
  <br>
  <em>Figure 1. Overall pipeline of HINT (Diagnosis–Time-series–Intervention).</em>
</div>

---

## 📊 Experimental Results

HINT is evaluated on the **MIMIC-III** dataset and compared against strong baselines, including LSTM-GNN and MTS-GCNN. It particularly improves performance on imbalanced metrics such as Macro AUPRC.

| Model            | Onset AUC | Wean AUC | Macro AUC | **Macro AUPRC** | **F1 Score** |
| ---------------- | :-------: | :------: | :-------: | :-------------: | :----------: |
| RF               |   87.5    |  98.9    |   81.6    |      43.9       |     52.4     |
| LSTM-GNN         |   84.4    |  98.7    |   85.2    |      48.0       |     60.6     |
| MTS-GCNN         | **89.9**  | **99.4** |   91.9    |      52.5       |     60.6     |
| **HINT (Proposed)** |   89.0    |  87.1    | **92.3** |   **75.2**      |   **69.8**   |

> **Highlight:** HINT achieves a **22.7 percentage-point improvement in Macro AUPRC** compared to the strongest baseline, demonstrating superior capability in detecting rare clinical transitions.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/eastlighting1/HINT.git
cd HINT
pip install -r requirements.txt
```

---

## 📂 Data Preparation (MIMIC-III)

This project uses the **MIMIC-III** database. You must have credentialed access via **PhysioNet**.

1. Download the MIMIC-III CSV files from PhysioNet.
2. Run the ETL pipeline via the unified entry point. This replaces the legacy `build_cohort.py` script:

   ```bash
   # Run the ETL pipeline (Extraction + Preprocessing)
   python src/hint/app/main.py mode=etl data.data_path=/path/to/mimic/raw
   ```

3. The pipeline generates:
   - Parquet files in `data/processed/`: static cohorts and time-series aggregates.
   - HDF5 files in `data/cache/`: windowed tensors (**VAL/MSK/DELTA**) for efficient training.

---

## 🧪 Usage

The application uses **Hydra** for configuration management. All major workflows are triggered via `src/hint/app/main.py` by switching the `mode`.

### 1. Train Automated ICD Coding Module

Train the MedBERT-based ICD encoder and the XGBoost ensemble stacker.

```bash
python src/hint/app/main.py mode=icd
```

### 2. Train Intervention Prediction Module (HINT)

Train the main CNN (TCN + ICD-Gating) model for intervention prediction. This is the default training workflow.

```bash
python src/hint/app/main.py mode=train \
  train.epochs=100 \
  train.batch_size=512
```

### 3. Run Tests & Coverage

We provide a dedicated test runner that executes unit, integration, and end-to-end tests and generates JaCoCo-style HTML coverage reports.  
The runner is configured via `configs/test_config.yaml`.

```bash
# Run all tests (default configuration)
python src/tests/runner.py

# Run only unit tests by overriding targets via CLI
python src/tests/runner.py test.targets="['src/tests/unit']"

# Run tests matching a keyword (e.g., "icd")
python src/tests/runner.py test.keywords="icd"
```

The HTML coverage report will be generated at:

```text
coverage_report/index.html
```

---

## 📁 Directory Structure

The project follows a **Domain-Driven Design (DDD)** architecture.

```text
HINT/
├── configs/                   # Hydra configuration files (hydra.yaml, cnn_config.yaml, test_config.yaml, etc.)
├── data/                      # Data storage (raw, processed, cache)
├── src/
│   ├── hint/                  # Main package
│   │   ├── app/               # Application Layer (entry point, factory)
│   │   ├── domain/            # Domain Layer (TrainableEntity)
│   │   ├── foundation/        # Foundation Layer (configs, DTOs, interfaces)
│   │   ├── infrastructure/    # Infrastructure Layer (networks, data source, registry)
│   │   └── services/          # Service Layer (orchestration logic)
│   │       ├── etl/           # ETL pipeline service
│   │       ├── icd_service.py # ICD training & XAI service
│   │       └── trainer.py     # Main CNN training service
│   │
│   └── tests/                 # Test suite
│       ├── unit/              # Unit tests (mock-heavy)
│       ├── integration/       # Integration tests (real I/O)
│       ├── e2e/               # End-to-end smoke tests
│       ├── utils/             # Test helpers (synthetic data generators)
│       └── runner.py          # Custom test runner with coverage support (Hydra-based)
│
├── requirements.txt
└── README.md
```

---

## 📜 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## 📝 Citation

If you find this work useful in your research, please cite the following thesis:

```bibtex
@mastersthesis{kim2026hint,
  author       = {Donghyeon Kim},
  title        = {Design and Implementation of a Clinical Decision Support System Using an Intervention Prediction Model},
  school       = {Gachon University},
  year         = {2026},
  month        = {February},
  type         = {Master's Thesis}
}
```

---

## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

**Donghyeon Kim**  
Email: your.email@example.com
GitHub: [https://github.com/username](https://github.com/username)
