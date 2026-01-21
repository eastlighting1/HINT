import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
import glob
import random
import re
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(BASE_DIR, "Report")

def ensure_report_dir():
    """Create the report directory if it does not exist."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        print(f"[Info] Created report directory: {REPORT_DIR}")

def analyze_parquet_file(file_path):
    """Generate a markdown report that summarizes a parquet file."""
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]
    report_path = os.path.join(REPORT_DIR, f"{base_name}_report.md")

    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema
        metadata = parquet_file.metadata
        table = parquet_file.read()
        df = None
        pandas_error = None

        def replace_string_view(pa_type):
            if pa.types.is_string_view(pa_type):
                return pa.string()
            if pa.types.is_list(pa_type):
                return pa.list_(replace_string_view(pa_type.value_type))
            if pa.types.is_large_list(pa_type):
                return pa.large_list(replace_string_view(pa_type.value_type))
            if pa.types.is_struct(pa_type):
                new_fields = [
                    pa.field(field.name, replace_string_view(field.type), field.nullable, field.metadata)
                    for field in pa_type
                ]
                return pa.struct(new_fields)
            return pa_type

        try:
            df = table.to_pandas()
        except Exception as e:
            pandas_error = str(e)
            try:
                new_fields = [
                    pa.field(field.name, replace_string_view(field.type), field.nullable, field.metadata)
                    for field in table.schema
                ]
                new_schema = pa.schema(new_fields, metadata=table.schema.metadata)
                df = table.cast(new_schema).to_pandas()
                pandas_error = None
            except Exception as e2:
                pandas_error = str(e2)
        
        def format_schema_text(parquet_schema):
            if hasattr(parquet_schema, "to_string"):
                schema_text = parquet_schema.to_string()
            else:
                schema_text = str(parquet_schema.to_arrow_schema())
            schema_text = re.sub(r"\n\s+child (\d+), element:", r"  child \1, element:", schema_text)
            return schema_text

        def sanitize_for_markdown(df_in, max_len=120):
            df_out = df_in.copy()
            obj_cols = df_out.select_dtypes(include=["object"]).columns
            for col in obj_cols:
                df_out[col] = df_out[col].map(lambda v: format_cell(v, max_len))
            return df_out

        def format_cell(value, max_len):
            if isinstance(value, (list, tuple, dict, set)):
                text = str(value)
            elif isinstance(value, (pa.Array,)):
                text = str(value.to_pylist())
            elif hasattr(value, "tolist"):
                try:
                    text = str(value.tolist())
                except Exception:
                    text = str(value)
            else:
                text = str(value)
            if len(text) > max_len:
                return text[: max_len - 3] + "..."
            return text

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Parquet Analysis Report: `{file_name}`\n\n")
            rel_path = os.path.relpath(file_path, ROOT_DIR)
            display_path = f"{os.path.basename(ROOT_DIR)}/{rel_path}"
            f.write(f"- **Path:** `{display_path}`\n")
            f.write(f"- **Created At:** {pd.Timestamp.now()}\n")
            f.write(f"- **Total Rows:** {table.num_rows:,}\n")
            f.write(f"- **Total Columns:** {len(schema.names)}\n")
            f.write(f"- **File Size:** {os.path.getsize(file_path) / (1024*1024):.2f} MB\n\n")

            f.write("## 1. File Metadata\n")
            f.write(f"- **Created By:** {metadata.created_by}\n")
            f.write(f"- **Format Version:** {metadata.format_version}\n")
            f.write(f"- **Num Row Groups:** {metadata.num_row_groups}\n\n")

            f.write("## 2. Schema Structure\n")
            f.write("```text\n")
            f.write(format_schema_text(schema))
            f.write("\n```\n\n")

            f.write("## 3. Column Summary\n")
            if df is None:
                f.write(f"*Pandas conversion skipped:* `{pandas_error}`\n\n")
            else:
                try:
                    summary_data = []
                    for col in df.columns:
                        col_type = df[col].dtype
                        try:
                            null_cnt = df[col].isnull().sum()
                        except Exception:
                            null_cnt = "N/A"
                        try:
                            unique_cnt = df[col].nunique()
                        except Exception:
                            unique_cnt = "N/A"
                        
                        summary_data.append({
                            "Column": col,
                            "Dtype": str(col_type),
                            "Null Count": null_cnt,
                            "Unique Count": unique_cnt
                        })
                    
                    df_summary = pd.DataFrame(summary_data)
                    f.write(df_summary.to_markdown(index=False))
                    f.write("\n\n")
                except Exception as e:
                    f.write(f"*Column summary failed:* `{e}`\n\n")

            f.write("## 4. Descriptive Statistics (Numeric)\n")
            if df is None:
                f.write("*Skipped because pandas conversion failed.*\n\n")
            else:
                try:
                    numeric_df = df.select_dtypes(include="number")
                    if numeric_df.empty:
                        f.write("*No numeric columns found.*\n\n")
                    else:
                        df_desc = numeric_df.describe().T
                        f.write(df_desc.to_markdown())
                        f.write("\n\n")
                except Exception as e:
                    f.write(f"*Descriptive statistics failed:* `{e}`\n\n")

            f.write("## 5. Data Samples\n")
            
            f.write("### 5.1 Head (First 5 rows)\n")
            if df is None:
                f.write("*Skipped because pandas conversion failed.*\n\n")
            else:
                try:
                    sample_df = sanitize_for_markdown(df.head(5))
                    f.write(sample_df.to_markdown(index=False))
                    f.write("\n\n")
                except Exception as e:
                    f.write(f"*Head sample failed:* `{e}`\n\n")

            f.write("### 5.2 Random Sample (3 rows)\n")
            if df is None:
                f.write("*Skipped because pandas conversion failed.*\n\n")
            else:
                try:
                    if len(df) > 3:
                        sample_df = sanitize_for_markdown(df.sample(3))
                        f.write(sample_df.to_markdown(index=False))
                    else:
                        sample_df = sanitize_for_markdown(df)
                        f.write(sample_df.to_markdown(index=False))
                    f.write("\n\n")
                except Exception as e:
                    f.write(f"*Random sample failed:* `{e}`\n\n")

        print(f"[Done] Report generated: {report_path}")

    except Exception as e:
        print(f"[Error] Failed to analyze {file_name}: {e}")

def main():
    """Find parquet files in the analyzer directory and generate reports."""
    ensure_report_dir()
    
    target_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    
    if not target_files:
        print("[Warning] No parquet files found in current directory.")
        return

    print(f"Found {len(target_files)} parquet files. Starting analysis...")
    for file_path in target_files:
        analyze_parquet_file(file_path)

if __name__ == "__main__":
    main()
