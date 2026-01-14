import h5py
import hdf5plugin
import json
import yaml
import random
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from contextlib import redirect_stdout
from transformers import AutoTokenizer

# 출력 포맷 설정
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "icd_config.yaml"

def load_config(config_path):
    default_config = {
        "data_dir": "data/cache",
        "split": "train",
        "tokenizer_path": "Charangan/MedBERT",
        "max_len": 32
    }
    if not config_path.exists():
        return default_config
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    for k, v in default_config.items():
        if k not in config:
            config[k] = v
    return config

def load_metadata(data_dir):
    meta = {}
    feat_path = data_dir / "feature_info.json"
    if feat_path.exists():
        with open(feat_path, "r") as f:
            meta.update(json.load(f))
            
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
            meta["stats"] = stats
            if "icd_classes" in stats:
                meta["id2label"] = {i: code for i, code in enumerate(stats["icd_classes"])}
    
    return meta

def find_key_insensitive(keys, target_candidates):
    keys_lower = {k.lower(): k for k in keys}
    for cand in target_candidates:
        if cand.lower() in keys_lower:
            return keys_lower[cand.lower()]
    return None

def inspect_hdf5_structure(f):
    print("\n" + "="*80)
    print("🏗️  HDF5 FILE STRUCTURE (Schema & Dtypes)")
    print("="*80)
    
    structure_info = []
    for key in f.keys():
        obj = f[key]
        info = {
            "Key Name": key,
            "Object Type": type(obj).__name__,
        }
        
        if isinstance(obj, h5py.Dataset):
            info["Shape"] = str(obj.shape)
            info["Dtype"] = str(obj.dtype)
            info["Chunks"] = str(obj.chunks) if obj.chunks else "None"
            info["Compression"] = str(obj.compression) if obj.compression else "None"
        
        structure_info.append(info)
    
    df = pd.DataFrame(structure_info)
    print(df.to_string(index=False))
    print("-" * 80)
    return structure_info

def analyze_label_distribution(f, meta):
    print("\n" + "="*80)
    print("📊 LABEL DISTRIBUTION ANALYSIS")
    print("="*80)

    y_key = find_key_insensitive(f.keys(), ['y'])
    if not y_key:
        print("❌ Label Key not found.")
        return

    # 데이터 로드
    y_data = f[y_key][:]
    total_samples = y_data.shape[0]
    id2label = meta.get("id2label", {})
    
    stats_list = []
    
    # 값 분석 (One-hot인지 Class Index인지 판단)
    unique_vals = np.unique(y_data)
    is_binary = np.all(np.isin(unique_vals, [0, 1]))
    has_ignore_index = np.any(unique_vals < 0) # -100 등

    # [FIX] 2차원이지만 값이 0/1이 아니거나 음수(Padding)가 포함된 경우 -> 시계열 클래스 인덱스로 처리
    if y_data.ndim == 2 and (not is_binary or has_ignore_index):
        print(f"Type: Time-Series Class Indices (Flattened) | Total Samples: {total_samples}")
        
        # Padding(-100) 제외하고 flatten
        y_flat = y_data.flatten()
        valid_mask = y_flat >= 0
        y_valid = y_flat[valid_mask]
        
        unique, counts = np.unique(y_valid, return_counts=True)
        total_valid = len(y_valid)
        
        for u, c in zip(unique, counts):
             # 4-Class 매핑 (Intervention)
            default_name = f"Class_{u}"
            if u == 0: default_name = "Stay Off (0)"
            elif u == 1: default_name = "Onset (1)"
            elif u == 2: default_name = "Stay On (2)"
            elif u == 3: default_name = "Wean (3)"

            stats_list.append({
                "Label ID": int(u),
                "Label Name": id2label.get(int(u), default_name),
                "Count": int(c),
                "Percentage": (c / total_valid) * 100 if total_valid > 0 else 0
            })

    # Case 1: Multi-label (One-hot encoding) - 순수 0/1로만 구성된 2D
    elif y_data.ndim == 2:
        print(f"Type: Multi-label (One-hot) | Total Samples: {total_samples}")
        class_counts = np.sum(y_data, axis=0)
        
        for idx, count in enumerate(class_counts):
            if count > 0:
                stats_list.append({
                    "Label ID": idx,
                    "Label Name": id2label.get(idx, f"Class_{idx}"),
                    "Count": int(count),
                    "Percentage": (count / total_samples) * 100
                })

    # Case 2: Multi-class (Indices) - 1D Array
    else:
        print(f"Type: Multi-class (Indices) | Total Samples: {total_samples}")
        unique, counts = np.unique(y_data, return_counts=True)
        
        for u, c in zip(unique, counts):
            stats_list.append({
                "Label ID": int(u),
                "Label Name": id2label.get(int(u), f"Class_{u}"),
                "Count": int(c),
                "Percentage": (c / total_samples) * 100
            })

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values(by="Count", ascending=False).reset_index(drop=True)
        
        print(f"Total Unique Labels Found: {len(df_stats)}")
        print("\n👉 Top 50 Frequent Labels:")
        print(df_stats.head(50).to_string(index=False))
    else:
        print("⚠️ No valid labels found (Check if all are padding).")
    
    print("-" * 80)

def inspect_sample_features(f, idx, meta, tokenizer):
    print(f"\n🔍 DETAILED SAMPLE INSPECTION (Index: {idx})")
    print("="*80)

    keys = list(f.keys())

    # [1] X_num
    num_key = find_key_insensitive(keys, ['x_num'])
    if num_key:
        print(f"\n[1] Numeric Features ('{num_key}')")
        data_tensor = f[num_key][idx] 
        print(f"   - Raw Shape: {data_tensor.shape}")
        
        # (Channels, Time) -> (291, 120)
        if data_tensor.ndim == 2:
            # 값이 있는(0이 아닌) 시점 찾기
            non_zero_timestamps = np.where(np.sum(np.abs(data_tensor), axis=0) > 0)[0]
            
            if len(non_zero_timestamps) > 0:
                rand_t = random.choice(non_zero_timestamps)
                data_vector = data_tensor[:, rand_t]
                print(f"   - ✅ Found {len(non_zero_timestamps)} valid time steps.")
                print(f"   - Valid Time Range: t={non_zero_timestamps[0]} ~ t={non_zero_timestamps[-1]}")
                print(f"   - Showing values at random VALID timestamp (t={rand_t})")
            else:
                rand_t = random.randint(0, data_tensor.shape[1] - 1)
                data_vector = data_tensor[:, rand_t]
                print(f"   - Showing values at random timestamp (t={rand_t})")
                print("   ⚠️ WARNING: This sample contains ALL ZEROS.")
        else:
            data_vector = data_tensor

        # 상위 10개 피처값만 출력
        df_num = pd.DataFrame({"Value": data_vector})
        non_zero_df = df_num[df_num["Value"] != 0]
        print(f"   - Non-zero Features at this step: {len(non_zero_df)} / {len(df_num)}")
        if not non_zero_df.empty:
            print(f"   👉 Top 10 Non-zero Values:")
            print(non_zero_df.head(10).to_string(index=True))

    # [2] X_cat
    cat_key = find_key_insensitive(keys, ['x_cat'])
    if cat_key:
        print(f"\n[2] Categorical Features ('{cat_key}')")
        print(f"   - Values: {f[cat_key][idx]}")

    # [3] Target Label (y)
    y_key = find_key_insensitive(keys, ['y'])
    if y_key:
        print(f"\n[3] Target Label ('{y_key}')")
        y_val = f[y_key][idx]
        
        # 시계열 출력 포맷 개선
        if y_val.ndim == 1 and len(y_val) > 1:
            # -100이 아닌 구간만 요약해서 보여줌
            valid_indices = np.where(y_val != -100)[0]
            if len(valid_indices) > 0:
                start, end = valid_indices[0], valid_indices[-1]
                print(f"   - Sequence Length: {len(y_val)}")
                print(f"   - Valid Range: t={start}~{end}")
                print(f"   - Values (valid): {y_val[start:end+1]}")
                
                # 상태 변화 감지
                changes = []
                for t in range(start+1, end+1):
                    if y_val[t] != y_val[t-1]:
                        changes.append(f"t={t}: {y_val[t-1]}->{y_val[t]}")
                if changes:
                    print(f"   - ⚠️ State Changes: {changes}")
                else:
                    print(f"   - No state changes (Constant state: {y_val[start]})")
            else:
                print(f"   - All values are Padding (-100)")
        else:
            print(f"   - Value: {y_val}")

    # [4] Binary Target (y_vent)
    vent_key = find_key_insensitive(keys, ['y_vent'])
    if vent_key:
        print(f"\n[4] Binary Target ('{vent_key}')")
        v_val = f[vent_key][idx]
        valid_indices = np.where(v_val != -100)[0]
        if len(valid_indices) > 0:
             print(f"   - Values (valid snippet): {v_val[valid_indices[0]:valid_indices[0]+10]} ...")

    print("="*80)

def main():
    config = load_config(CONFIG_PATH)
    data_dir = PROJECT_ROOT / Path(config.get("data_dir", "data/cache"))
    split_name = config.get("split", "train")
    
    # Intervention 파일 우선 검색
    h5_path = data_dir / f"train_intervention_{split_name}.h5"
    
    output_txt_path = data_dir / f"train_intervention_{split_name}_report.txt"

    if not h5_path.exists():
        import glob
        files = glob.glob(str(data_dir / "*.h5"))
        if files:
            h5_path = Path(files[0])
            print(f"⚠️ Configured file not found. Inspecting: {h5_path.name}")
        else:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {h5_path}")
            return

    meta = load_metadata(data_dir)
    print(f"🚀 Analyzing {h5_path.name}...")
    print(f"📄 Report will be saved to: {output_txt_path}")

    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        with redirect_stdout(out_f):
            with h5py.File(h5_path, "r") as f:
                inspect_hdf5_structure(f)
                analyze_label_distribution(f, meta)
                
                # 랜덤 샘플 확인
                sample_key = find_key_insensitive(f.keys(), ['x_num', 'x'])
                if sample_key:
                    total_samples = f[sample_key].shape[0]
                    idx = random.randint(0, total_samples - 1)
                    inspect_sample_features(f, idx, meta, None)
                else:
                    print("❌ 샘플링할 메인 키를 찾을 수 없습니다.")

    print("✅ Analysis Complete! Check the txt file.")

if __name__ == "__main__":
    main()
