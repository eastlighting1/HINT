import h5py
import numpy as np
import pandas as pd
import os
import glob
import sys
from datetime import datetime

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "cache")
REPORT_DIR = os.path.join(BASE_DIR, "Report")

def ensure_report_dir():
    """Create the report directory if it does not exist."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

def get_attributes(obj):
    """Return HDF5 object attributes as a dictionary."""
    attrs = {}
    for k, v in obj.attrs.items():
        if isinstance(v, bytes):
            try:
                attrs[k] = v.decode('utf-8')
            except:
                attrs[k] = str(v)
        else:
            attrs[k] = v
    return attrs

def inspect_dataset(name, dataset, f_out):
    """Write a detailed dataset analysis section to the report."""
    f_out.write(f"### Dataset: `{name}`\n")
    
    info = {
        "Shape": dataset.shape,
        "Dtype": dataset.dtype,
        "Chunks": dataset.chunks,
        "Compression": dataset.compression
    }
    f_out.write("- **Metadata:**\n")
    for key, value in info.items():
        f_out.write(f"  - {key}: {value}\n")
    
    attrs = get_attributes(dataset)
    if attrs:
        f_out.write(f"- **Attributes:** {attrs}\n")

    try:
        data = dataset[:]
        
        if data.ndim == 0:
            f_out.write(f"- **Value:** {data}\n\n")
            return

        if np.issubdtype(dataset.dtype, np.number):
            stats = {
                "Min": np.min(data),
                "Max": np.max(data),
                "Mean": np.mean(data),
                "Std": np.std(data)
            }
            if np.isnan(stats["Mean"]):
                 stats = {k: "NaN included" for k in stats}
            
            f_out.write("- **Statistics:**\n")
            for key, value in stats.items():
                if hasattr(value, "item"):
                    value = value.item()
                f_out.write(f"  - {key}: {value}\n")

        if data.ndim == 1:
            df = pd.DataFrame(data, columns=['Value'])
        elif data.ndim == 2:
            cols = [f"Col_{i}" for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=cols)
        else:
            f_out.write(f"- **Sample (Flattened First 10):** {data.flatten()[:10]}\n")
            f_out.write("\n---\n")
            return

        f_out.write("\n**Sample Data (Top 5 rows):**\n")
        f_out.write(df.head(5).to_markdown())
        f_out.write("\n")

    except Exception as e:
        f_out.write(f"- **Error reading data:** {str(e)}\n")
    
    f_out.write("\n---\n")

def visit_item(name, obj, f_out):
    """Visit HDF5 objects and write their summaries to the report."""
    if isinstance(obj, h5py.Dataset):
        inspect_dataset(name, obj, f_out)
    elif isinstance(obj, h5py.Group):
        f_out.write(f"### Group: `{name}`\n")
        attrs = get_attributes(obj)
        if attrs:
            f_out.write(f"- **Attributes:** {attrs}\n")
        f_out.write(f"- **Members:** {list(obj.keys())}\n")
        f_out.write("\n---\n")

def analyze_h5_file(file_path):
    """Generate a markdown report that summarizes an HDF5 file."""
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    report_path = os.path.join(REPORT_DIR, f"{base_name}_report.md")

    try:
        with h5py.File(file_path, 'r') as f:
            with open(report_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f"# HDF5 Analysis Report: `{file_name}`\n\n")
                f_out.write(f"- **Path:** `{file_path}`\n")
                f_out.write(f"- **Created At:** {datetime.now()}\n")
                f_out.write(f"- **File Size:** {os.path.getsize(file_path) / (1024*1024):.2f} MB\n")
                
                root_attrs = get_attributes(f)
                if root_attrs:
                    f_out.write(f"- **Global Attributes:** {root_attrs}\n")
                
                f_out.write("\n## Structure & Content Analysis\n\n")
                
                f.visititems(lambda name, obj: visit_item(name, obj, f_out))

        print(f"[Done] Report generated: {report_path}")

    except Exception as e:
        print(f"[Error] Failed to analyze {file_name}: {e}")

def main():
    """Find HDF5 files in the analyzer directory and generate reports."""
    ensure_report_dir()
    
    target_files = glob.glob(os.path.join(DATA_DIR, "*.h5"))
    
    if not target_files:
        print("[Warning] No .h5 files found in current directory.")
        return

    print(f"Found {len(target_files)} h5 files. Starting analysis...")
    for file_path in target_files:
        analyze_h5_file(file_path)

if __name__ == "__main__":
    main()
