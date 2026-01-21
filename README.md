# HINT: Hierarchical ICD-aware Network for Time-series Intervention

<div align="center">

![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=hydra&logoColor=white)
![uv](https://img.shields.io/badge/Managed%20by-uv-purple?style=for-the-badge)

**A Hierarchical Clinical Decision Support System (CDSS) for ICU Mechanical Ventilation Prediction**

[Introduction](#introduction) • [Core Architecture](#core-architecture) • [Quick Start](#quick-start-with-uv) • [Usage Guide](#usage-guide) • [Project Layout](#project-layout) • [Benchmarks](#benchmarks--performance)

</div>

---

## Introduction

Welcome to the **HINT** repository!

**HINT** stands for *Hierarchical ICD-aware Network for Time-series Intervention*. It is a cutting-edge Clinical Decision Support System (CDSS) developed to assist clinicians in the Intensive Care Unit (ICU) by predicting the need for **mechanical ventilation interventions**.

### Why is this important?

In the ICU, a patient's condition changes rapidly. Clinicians must process massive amounts of data—vital signs (heart rate, SpO2) and lab results—in real-time. However, existing AI models often fail to connect these "low-level" signals with the patient's "high-level" diagnosis (ICD codes), leading to suboptimal predictions.

**HINT solves this by thinking like a doctor:** it first understands the patient's underlying diagnosis (even if records are incomplete!) and then uses that context to interpret the fluctuating vital signs more accurately.

### Key Features at a Glance

- **Hierarchical Thinking**: A two-stage pipeline that separates *diagnosis inference* from *event prediction*, mimicking clinical reasoning.
- **Robust to Noisy Data**: Uses **Partial Label Learning** with MedBERT. Even if the medical records are messy or incomplete, HINT can infer the probable diagnosis.
- **Context-Aware**: It doesn't just look at the numbers. An **ICD-Conditioned Gating Mechanism** dynamically adjusts which vital signs are most important based on the patient's specific disease.
- **Imbalance Expert**: Designed for rare events. It uses specialized loss functions (Focal Loss) to accurately predict critical moments like `Onset` (starting ventilation) or `Weaning` (stopping it).
- **Explainable (XAI)**: It's not a black box. HINT provides **SHAP** and **LIME** analyses so clinicians can understand *why* a prediction was made.

---

## Core Architecture

The system is built upon a **Domain-Driven Design (DDD)** architecture, ensuring that the code is as robust as the model itself.

### Stage 1: Automated ICD Coding Module

Before looking at the time-series, HINT understands the patient.

- **Input**: Static data (age, gender) & Candidate ICD-9 code sets.
- **Tech Stack**: `MedBERT` (Text Encoder) + `XGBoost` (Stacking).
- **Output**: A dense "Context Vector" representing the patient's admission-level diagnosis.

### Stage 2: Intervention Prediction Module (GFINet)

This is where the magic happens. The model combines the context from Stage 1 with real-time monitoring data.

- **Input**:
  - **Time-Series Tensor**: (Values, Mask, Time-Delta) for 30+ vital signs.
  - **Context Vector**: Output from Stage 1.
- **Tech Stack**:
  - **Group-wise TCN**: Analyzes temporal patterns efficiently.
  - **Gating Mechanism**: Fuses diagnosis context to re-weight time-series features.
- **Output**: Probabilities for 4 states (`ONSET`, `WEAN`, `STAY ON`, `STAY OFF`).

<div align="center">
  <img width="850" alt="HINT Architecture Diagram" src="https://github.com/user-attachments/assets/85fde6e4-3800-4bd2-8983-e33a504cd72d" />
  <br>
  <em>Figure 1. The comprehensive pipeline: From raw MIMIC-III data to actionable clinical insights.</em>
</div>

---

## Quick Start (with `uv`)

We use **[uv](https://github.com/astral-sh/uv)**, a blazing fast Python package manager. If you haven't used it before, you'll love the speed!

### 1. Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (Highly recommended for GPU acceleration)
- **MIMIC-III Access**: You need credentialed access from [PhysioNet](https://physionet.org/content/mimiciii/).

### 2. Installation

Clone the repo and let `uv` handle the rest. No need to manually create virtual environments.

```bash
# 1. Clone the repository
git clone https://github.com/eastlighting1/HINT.git
cd HINT

# 2. Sync dependencies (This creates a virtualenv and installs everything)
uv sync
```

### 3. Data Setup

Since we cannot distribute MIMIC-III data, please place your downloaded CSV files in a folder.

```bash
# Example directory structure
mkdir -p data/raw
# Place ADMISSIONS.csv, CHARTEVENTS.csv, etc. inside data/raw
```

-----

## Usage Guide

HINT is configured using **Hydra**. This allows you to orchestrate the entire pipeline via the command line interface (CLI) exposed by `src/hint/app/main.py`.

The basic command structure is:

```bash
uv run hint mode=[etl|icd|intervention|all] [overrides]
```

### 1. ETL Pipeline (Data Processing)

Cleans the raw CSV data and generates HDF5 tensors.

```bash
uv run hint mode=etl data.raw_dir="./data/raw"
```

> **Output**: Processed artifacts are saved in `data/processed/` and `data/cache/` by default.

### 2. Diagnosis Inference (Stage 1)

Trains the ICD coding module to learn patient representations.

```bash
uv run hint mode=icd
```

### 3. Intervention Prediction (Stage 2)

Trains the intervention prediction model using the output from Stage 1. You can dynamically override hyperparameters.

```bash
uv run hint mode=intervention \
    intervention.epochs=50 \
    intervention.batch_size=256 \
    intervention.lr=0.0003
```

### 4. Full Pipeline

Executes the complete workflow sequentially (ETL -> ICD -> intervention).

```bash
uv run hint mode=all
```

-----

## Project Layout

Key directories you will interact with during development and runs:

```text
HINT/
├── configs/                 # Hydra configuration files
├── data/
│   ├── raw/                 # Raw MIMIC-III CSV files
│   ├── processed/           # ETL outputs (Parquet)
│   └── cache/               # Cached HDF5 tensors
├── resources/               # ICD/variable metadata
├── artifacts/               # Trained model artifacts
├── outputs/                 # Logs and run outputs
└── Analyzer/                # Data inspection utilities
```

You can review and override defaults in `configs/config.yaml`, `configs/etl_config.yaml`, `configs/icd_config.yaml`, and `configs/intervention_config.yaml`.

-----

## Benchmarks & Performance

HINT has been rigorously evaluated on the MIMIC-III dataset. It specifically excels in the **Macro AUPRC** metric, which is the most critical metric for imbalanced medical data.

| Model Architecture | Macro AUC | **Macro AUPRC** | F1 Score |
| :----------------- | :-------: | :-------------: | :------: |
| **Random Forest**  |   81.6    |      43.9       |   52.4   |
| **LSTM-GNN**       |   85.2    |      48.0       |   60.6   |
| **MTS-GCNN**       |   91.9    |      52.5       |   60.6   |
| **HINT (Ours)**    | **92.3**  |   **75.2**      | **69.8** |

> **Result:** HINT improves the AUPRC by **+22.7%** compared to the strongest baseline (MTS-GCNN). This means significantly fewer false alarms for clinicians.

-----

## Code Structure

We follow a **DDD (Domain-Driven Design)** pattern to keep things organized.

```text
src/
├── hint/
│   ├── app/                       # Entry points and app wiring
│   │   ├── main.py                # CLI entry point
│   │   └── factory.py             # Service composition
│   ├── domain/                    # Core entities and value objects
│   ├── foundation/                # DTOs, configs, and shared interfaces
│   ├── infrastructure/            # Models, data sources, telemetry
│   └── services/                  # Execution pipelines
│       ├── etl/                   # Data preprocessing
│       └── training/
│           ├── automatic_icd_coding/   # Stage 1 training
│           ├── predict_intervention/   # Stage 2 training
│           └── common/                 # Shared training utilities
```

-----

## 📝 Citation

If this work helps your research, please cite the Master's Thesis:

```bibtex
@article{kim2026design,
title = {Design and Implementation of a Clinical Decision Support System Using an Intervention Prediction Model},
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

-----

## 📮 Contact & Support

We love hearing from the community\! If you have questions, run into issues, or just want to discuss medical AI:

- **Author**: Donghyeon Kim
- **Email**: eastlighting1@gachon.ac.kr
- **GitHub Issues**: Please open an issue if you find a bug\!

Happy Coding\! 🚀
