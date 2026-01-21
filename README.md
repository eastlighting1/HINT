# HINT: Hierarchical ICD-aware Network for Time-series Intervention

<div align="center">

![Version](https://img.shields.io/badge/Version-0.1.0-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=hydra&logoColor=white)
![uv](https://img.shields.io/badge/Managed%20by-uv-purple?style=for-the-badge)

**A Hierarchical Clinical Decision Support System (CDSS) for ICU Mechanical Ventilation Prediction**

[Introduction](#introduction) • [Core Architecture](#core-architecture) • [Quick Start](#quick-start-with-uv) • [Usage Guide](#usage-guide) • [Project Layout](#project-layout) • [Data Analysis Tools](#data-analysis-tools) • [Benchmarks](#benchmarks--performance) • [Code Structure](#code-structure)

</div>

---

## Introduction

Welcome to the **HINT** repository!

**HINT** stands for *Hierarchical ICD-aware Network for Time-series Intervention*. It is a cutting-edge Clinical Decision Support System (CDSS) developed to assist clinicians in the Intensive Care Unit (ICU) by predicting the need for **mechanical ventilation interventions**.

### Why is this important?

In the ICU, a patient's condition changes rapidly. Clinicians must process massive amounts of data — vital signs (heart rate, SpO2) and lab results — in real-time. However, existing AI models often fail to connect these "low-level" signals with the patient's "high-level" diagnosis (ICD codes), leading to suboptimal predictions.

**HINT solves this by thinking like a doctor:** it first understands the patient's underlying diagnosis (even if records are incomplete!) and then uses that context to interpret the fluctuating vital signs more accurately.

### Key Features at a Glance

- **Hierarchical Pipeline**: ETL builds tensors, ICD coding learns context, and intervention prediction models ventilation states.
- **Partial-Label ICD Learning**: Uses CLPL/adaptive CLPL losses with candidate ICD-9 sets to handle incomplete labels.
- **Context-Aware Prediction**: Optional ICD-conditioned gating reweights time-series features during intervention prediction.
- **Imbalance Handling**: Focal loss and weighted sampling target rare ventilation transitions.
- **Hydra-Driven**: All stages are configured via `configs/*.yaml` with CLI overrides.

---

## Core Architecture

The system is built upon a **Domain-Driven Design (DDD)** architecture, ensuring that the code is as robust as the model itself.

### Stage 1: Automated ICD Coding Module

HINT first learns admission-level ICD context from the ETL tensors.

- **Input**: Time-series tensors (values/mask/delta), static categorical features, and candidate ICD-9 code sets.
- **Tech Stack**:
  - **Dense + cross feature modeling** for ICD coding.
  - **Candidate-set aware loss (CLPL/adaptive CLPL)** for partial-label learning.
- **Output**: An admission-level ICD context vector that can be injected into the intervention dataset.

### Stage 2: Intervention Prediction Module (TCN)

This is where the magic happens. The model combines the context from Stage 1 with real-time monitoring data.

- **Input**:
  - **Time-Series Tensor**: (Values, Mask, Time-Delta) for 30+ vital signs.
  - **Context Vector**: Output from Stage 1.
- **Tech Stack**:
  - **Temporal Convolutional Network (TCN)** for time-series modeling.
  - **ICD Gating (optional)** to modulate numeric features using the ICD context.
- **Output**: Probabilities for 4 states (`ONSET`, `WEAN`, `STAY ON`, `STAY OFF`).

<br>
<div align="center">
  <img width="850" alt="HINT Architecture Diagram" src="https://github.com/user-attachments/assets/0279bdfa-4cdb-44a4-a0e2-97812ab62c85" />
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
# Place ADMISSIONS.csv(.gz), CHARTEVENTS.csv(.gz), etc. inside data/raw
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

> **Note:** The ICD stage also generates `X_icd` features for the intervention dataset (feature injection).
> By default, only `DCNv2` runs. To try multiple backbones, set `icd.model_testing=true` and edit `icd.models_to_run` in `configs/icd_config.yaml`.

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

## Data Analysis Tools

For quick inspection of generated data, use the Analyzer utilities:

```bash
# HDF5 structure and tensor stats
uv run Analyzer/h5_analyzer.py

# Parquet schema and column stats
uv run Analyzer/parquet_analyzer.py
```

Reports are saved under `Analyzer/Report/`.

-----

## Benchmarks & Performance

HINT has been rigorously evaluated on the MIMIC-III dataset. It specifically excels in the **Macro AUPRC** metric, which is the most critical metric for imbalanced medical data.

| Model Architecture | Macro AUC | **Macro AUPRC** | F1 Score |
| :----------------- | :-------: | :-------------: | :------: |
| **Random Forest**  |   81.6    |      43.9       |   52.4   |
| **LSTM-GNN**       |   85.2    |      48.0       |   60.6   |
| **MTS-GCNN**       |   91.9    |      52.5       |   60.6   |
| **HINT (Ours)**    | **92.3**  |   **75.2**      | **69.8** |

> **Result:** HINT improves the AUPRC by **+22.7%** compared to the strongest baseline (MTS-GCNN). This means significantly fewer false alarms for clinicians. Reported results require MIMIC-III access to reproduce.

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

## Citation

If this work helps your research, please cite the paper :

```bibtex
@article{HINT,
title = {HINT: Hierarchical ICD-aware Network for Time-series Intervention},
journal = {TBD},
year = {2026},
author = {Donghyeon Kim},
}
```

-----

## Contact & Support

We love hearing from the community\! If you have questions, run into issues, or just want to discuss medical AI:

- **Author**: Donghyeon Kim
- **Email**: eastlighting1@gachon.ac.kr
- **GitHub Issues**: Please open an issue if you find a bug\!

Happy coding.
