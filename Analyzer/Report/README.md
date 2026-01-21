# Data Analysis Tools

This directory contains Python utilities designed to automatically inspect, analyze, and generate Markdown reports for HDF5 and Parquet data files.

## Contents

| File | Description | Target Input |
|------|-------------|--------------|
| `h5_analyzer.py` | Traverses HDF5 file structures (Groups/Datasets), extracts attributes, and calculates statistics for numeric arrays. | `.h5` files in `../data/cache/` |
| `parquet_analyzer.py` | Inspects Parquet schemas/metadata, calculates column-wise statistics (nulls, unique values), and provides data samples. | `.parquet` files in `../data/processed/` |

## Directory Structure Requirement

The scripts depend on a specific relative path structure to locate input files. Ensure your project follows this layout:

~~~text
HINT/
├── data/
│   ├── cache/          # Place raw .h5 files here
│   └── processed/      # Place .parquet files here
└── Analyzer/           # (Current Directory)
    ├── h5_analyzer.py
    ├── parquet_analyzer.py
    └── Report/         # Generated Markdown reports will be saved here
~~~

## Usage

### Analyzing HDF5 Files

Run the script to process all `.h5` files located in the `data/cache` directory.

~~~bash
uv run h5_analyzer.py
~~~

**Output:** A structured report including dataset shapes, types, compression settings, attributes, and basic statistics (min/max/mean/std).

---

### Analyzing Parquet Files

Run the script to process all `.parquet` files located in the `data/processed` directory.

~~~bash
uv run parquet_analyzer.py
~~~

**Output:** A comprehensive report including file metadata, full schema, column summary (types, null counts), descriptive statistics, and head/random data samples.

## Report Output

All generated reports are saved in the `Report/` subdirectory within this folder.

* **Format:** Markdown (`.md`)
* **Naming:** `<original_filename>_report.md`
