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
   <br>
   <img width="800" alt="HINT Architectur" src="https://github.com/user-attachments/assets/85fde6e4-3800-4bd2-8983-e33a504cd72d" />


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

- Python 3.9+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)
- [Polars](https://pola.rs/) (for fast data processing)

### Installation

```bash
git clone https://github.com/eastlighting1/HINT.git
cd HINT
pip install -r requirements.txt
```

-----

## 📂 Data Preparation (MIMIC-III)

This project uses the **MIMIC-III** database. You must have credentialed access via **PhysioNet**.

1.  Download the MIMIC-III CSV files from PhysioNet.

2.  Run the **ETL Pipeline** to extract cohorts, process time-series, and generate windowed tensors.

    ```bash
    # Run the full ETL pipeline (Extraction -> Processing -> Tensor Conversion)
    # Ensure 'data.raw_dir' points to your downloaded MIMIC CSVs
    python src/hint/app/main.py mode=etl data.raw_dir=/path/to/mimic/raw
    ```

3.  The pipeline generates artifacts in the directory specified by `logging.artifacts_dir` (default: `artifacts/`):

      - **Dataframes** (`artifacts/data/`): `patients.parquet`, `interventions.parquet`, `dataset_123.parquet`.
      - **Tensors** (`artifacts/data/`): `train.h5`, `val.h5`, `test.h5` (Windowed & Normalized).
      - **Metadata** (`artifacts/metrics/`): `train_stats.json`, `feature_info.json`.

-----

## 🧪 Usage

The application uses **Hydra** for configuration management. Workflows are orchestrated via `src/hint/app/main.py` by switching the `mode`.

### 1. Train Automated ICD Coding Module

Trains the **MedBERT-based ICD encoder** and the XGBoost ensemble stacker using the processed parquet data.

```bash
python src/hint/app/main.py mode=icd
```

### 2. Train Intervention Prediction Module (CNN)

Trains the main **GFINet (CNN)** model for intervention prediction using the HDF5 tensor streams. This mode includes **Training**, **Temperature Calibration**, and **Evaluation**.

```bash
python src/hint/app/main.py mode=cnn \
  cnn.model.epochs=100 \
  cnn.model.batch_size=512
```

### 3. Full Training Pipeline

Execute both ICD training and CNN training sequentially.

```bash
python src/hint/app/main.py mode=train
```

### 4. Run Tests & Coverage (TBD)

We provide a dedicated test runner that executes unit, integration, and end-to-end tests.

```bash
# Run all tests (default configuration)
python src/test/runner.py

# Run only unit tests
python src/test/runner.py test.targets="['src/test/unit']"

# Run tests matching a keyword
python src/test/runner.py test.keywords="icd"
```

The HTML coverage report will be generated at `coverage_report/index.html`.

-----

## 📁 Directory Structure

The project follows a rigorous **Domain-Driven Design (DDD)** architecture to ensure separation of concerns between data processing, model logic, and infrastructure.

```text
src/hint/
├── app/                        # Application Layer
│   ├── main.py                 # Entry Point & Pipeline Orchestrator
│   └── factory.py              # Dependency Injection Factory
│
├── domain/                     # Domain Layer (Business Logic & State)
│   ├── entities.py             # TrainableEntity (Model State, EMA, Steps)
│   └── vo.py                   # Value Objects (Immutable Configs)
│
├── foundation/                 # Foundation Layer (Shared Kernels)
│   ├── configs.py              # Config Loading & Validation
│   ├── dtos.py                 # Data Transfer Objects (TensorBatch)
│   ├── exceptions.py           # Custom Domain Exceptions
│   └── interfaces.py           # Abstract Base Classes (Port Definitions)
│
├── infrastructure/             # Infrastructure Layer (Adapters)
│   ├── datasource.py           # Streaming Sources (HDF5, Parquet Adapters)
│   ├── registry.py             # Artifact Persistence (File I/O)
│   ├── telemetry.py            # Logging & Metrics (Rich Observer)
│   ├── networks.py             # PyTorch Modules (GFINet, MedBERT)
│   └── components.py           # Shared ML Components (FocalLoss, TempScaler)
│
└── services/                   # Service Layer (Use Cases)
    ├── etl/                    # ETL Pipeline
    │   ├── service.py          # ETL Orchestrator
    │   └── components/         # Pipeline Steps (Static, TimeSeries, Tensor...)
    ├── icd/                    # ICD Domain Service
    │   └── service.py          # Training, Stacking, XAI logic
    └── training/               # Intervention Domain Service
        ├── trainer.py          # CNN Training Loop
        └── evaluator.py        # Calibration & Evaluation
```

## 📜 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## 📝 Citation

If you find this work useful in your research, please cite the following thesis:

```bibtex
@article{kim2026design,
title = {HINT : Hierarchical ICD-aware Network for Time-series Intervention},
journal = {TBD},
volume = {},
pages = {},
year = {2026},
issn = {},
doi = {},
url = {},
author = {Donghyeon Kim},
keywords = {ICD Coding, Partial-label learning, Intervention Prediction, Clinical Decision Support System, TCN, Class Imbalance, Explainable AI, MIMIC-III},
abstract = {Intensive Care Units generate massive volumes of heterogeneous clinical data from electronic medical records and bedside monitoring, yet the rapid and unstable nature of critical illness makes it difficult for clinicians to integrate these signals in real time. Building trustworthy clinical decision support systems (CDSSs) therefore requires models that can handle irregular, sparse multivariate time series and incorporate high-level diagnostic context. In practice, ICU time series are heavily affected by missingness and non-uniform sampling, while International Classification of Diseases codes exhibit incomplete and overlapping label structures together with extreme class imbalance. To address these challenges, this study proposes a hierarchical diagnosis-time-series-intervention CDSS for ICU intervention prediction. The system consists of a two-stage pipeline. First, an Automated ICD Coding module infers an admission-level representative ICD label from ICD-9 candidate sets and static/numerical patient features. The module embeds realistic documentation uncertainty through partial-label learning, and it mitigates extreme imbalance via inverse-frequency sampling and class-balanced focal loss, yielding robust diagnostic context under weak supervision. Second, an Intervention Prediction module constructs ICU time-series tensors augmented with the inferred ICD context and predicts four mechanical ventilation states. The proposed model combines (i) multi-branch TCN-based CNN blocks that learn feature-group-specific temporal representations to reduce representation interference, and (ii) an ICD-driven gating mechanism that dynamically reweights numerical time-series features according to diagnostic context, enabling context-adaptive intervention inference. To ensure clinical interpretability, SHAP-based global explanations and LIME-based local explanations are provided in parallel for both modules. Experiments on MIMIC-III demonstrate that the proposed system achieves the best balanced performance under severe imbalance, reaching a Macro AUPRC of 75.2% and an F1 score of 69.8%, while also delivering a Macro AUC of 92.3% comparable to or slightly surpassing strong hybrid baselines. ICD impact analyses further confirm that clinically valid ICD context consistently outperforms random ICD injection, and that admission-level diagnostic context is more effective than ICU-stay-level fixation. These results validate the proposed hierarchical CDSS as a practical and imbalance-robust framework for context-aware ICU intervention prediction.}
}
```

---

## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

**Donghyeon Kim**  
Email: eastlighting1@gachon.ac.kr <br>
GitHub: [https://github.com/eastlighting1](https://github.com/eastlighting1)
