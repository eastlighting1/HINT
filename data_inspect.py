import h5py
import hdf5plugin
import json
import yaml
import random
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from contextlib import redirect_stdout  # [추가] 출력을 파일로 돌리기 위해 사용
from transformers import AutoTokenizer

# 출력 포맷 설정
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# ==========================================
# [설정] 프로젝트 구조 반영
# ==========================================
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
    
    # 1. feature_info.json
    feat_path = data_dir / "feature_info.json"
    if feat_path.exists():
        with open(feat_path, "r") as f:
            meta.update(json.load(f))
            
    # 2. stats.json
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
            meta["stats"] = stats
            if "icd_classes" in stats:
                # {0: 'A00', 1: 'B01', ...} 형태의 맵 생성
                meta["id2label"] = {i: code for i, code in enumerate(stats["icd_classes"])}
    
    return meta

def find_key_insensitive(keys, target_candidates):
    """
    대소문자 구분 없이 키를 찾습니다.
    예: keys=['X_num'], target_candidates=['x_num'] -> returns 'X_num'
    """
    keys_lower = {k.lower(): k for k in keys}
    for cand in target_candidates:
        if cand.lower() in keys_lower:
            return keys_lower[cand.lower()]
    return None

def inspect_hdf5_structure(f):
    """HDF5 파일 내부 구조 출력"""
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

# [추가] 라벨 분포 분석 함수
def analyze_label_distribution(f, meta):
    """
    전체 데이터셋의 라벨 분포를 분석하여 출력합니다.
    """
    print("\n" + "="*80)
    print("📊 LABEL DISTRIBUTION ANALYSIS")
    print("="*80)

    y_key = find_key_insensitive(f.keys(), ['y', 'label', 'target'])
    if not y_key:
        print("❌ Label Key not found.")
        return

    # 데이터 로드 (메모리 주의: 너무 크면 chunk로 처리해야 함. 여기선 전체 로드 가정)
    y_data = f[y_key][:] 
    total_samples = y_data.shape[0]
    id2label = meta.get("id2label", {})
    
    stats_list = []

    # Case 1: Multi-label (One-hot encoding, 2D array)
    if y_data.ndim == 2:
        print(f"Type: Multi-label (One-hot) | Total Samples: {total_samples}")
        # 각 클래스별 합계 (빈도)
        class_counts = np.sum(y_data, axis=0)
        
        for idx, count in enumerate(class_counts):
            if count > 0: # 등장한 라벨만 표시
                stats_list.append({
                    "Label ID": idx,
                    "Label Name": id2label.get(idx, f"Class_{idx}"),
                    "Count": int(count),
                    "Percentage": (count / total_samples) * 100
                })

    # Case 2: Multi-class (Indices, 1D array)
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

    # DataFrame 생성 및 정렬
    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values(by="Count", ascending=False).reset_index(drop=True)
        
        print(f"Total Unique Labels Found: {len(df_stats)}")
        print("\n👉 Top 50 Frequent Labels:")
        print(df_stats.head(50).to_string(index=False))
        
        print("\n👉 Bottom 10 Rare Labels:")
        print(df_stats.tail(10).to_string(index=False))
    else:
        print("⚠️ No labels found or all counts are zero.")
    
    print("-" * 80)


def inspect_sample_features(f, idx, meta, tokenizer):
    """샘플 상세 분석"""
    print(f"\n🔍 DETAILED SAMPLE INSPECTION (Index: {idx})")
    print("="*80)

    keys = list(f.keys())

    # 1. 수치형 데이터 (X_num)
    num_key = find_key_insensitive(keys, ['x_num', 'x_static', 'numerical', 'static'])
    
    if num_key:
        print(f"\n[1] Numeric Features ('{num_key}')")
        data_tensor = f[num_key][idx] 
        print(f"   - Raw Shape: {data_tensor.shape}")
        
        if data_tensor.ndim == 2:
            data_vector = data_tensor[:, -1] 
            print("   - Showing values at last timestamp (t=-1)")
        else:
            data_vector = data_tensor

        feat_names = (
            meta.get("feat_names_numeric") or 
            meta.get("base_feats_numeric") or 
            [f"Feature_{i}" for i in range(len(data_vector))]
        )
        
        min_len = min(len(data_vector), len(feat_names))
        
        df_num = pd.DataFrame({
            "Index": range(min_len),
            "Feature Name": feat_names[:min_len],
            "Value (Last Step)": data_vector[:min_len],
        })
        
        non_zero_df = df_num[df_num["Value (Last Step)"] != 0]
        print(f"   - Non-zero Features: {len(non_zero_df)} / {len(df_num)}")
        
        if not non_zero_df.empty:
            print(f"   👉 Top 20 Non-zero Values:")
            print(non_zero_df.head(20).to_string(index=False))
        else:
            print("   👉 All values are zero (Sparse/Masked).")
    else:
        print("\n[1] Numeric Features: Not found.")

    # 2. 범주형 데이터 (X_cat)
    cat_key = find_key_insensitive(keys, ['x_cat', 'categorical'])
    if cat_key:
        print(f"\n[1-2] Categorical Features ('{cat_key}')")
        cat_data = f[cat_key][idx]
        print(f"   - Shape: {cat_data.shape}")
        if cat_data.ndim == 2:
            print(f"   - Last Timestep Values: {cat_data[:, -1]}")
        else:
            print(f"   - Values: {cat_data}")

    # 3. 텍스트 데이터 (input_ids)
    input_key = find_key_insensitive(keys, ['input_ids', 'x_text', 'text'])
    mask_key = find_key_insensitive(keys, ['attention_mask', 'mask'])
    
    if input_key:
        print(f"\n[2] Text Sequence Features ('{input_key}')")
        token_ids = f[input_key][idx]
        attn_mask = f[mask_key][idx] if mask_key else np.ones_like(token_ids)
        
        print(f"   - Shape: {token_ids.shape}")
        
        if tokenizer:
            valid_ids = [t for t, m in zip(token_ids, attn_mask) if m == 1]
            decoded_text = tokenizer.decode(valid_ids)
            print(f"   - 📝 Decoded Content: \"{decoded_text}\"")
        else:
            print(f"   - Raw IDs: {token_ids}")

    # 4. 타겟 라벨 (y)
    y_key = find_key_insensitive(keys, ['y', 'label', 'target'])
    if y_key:
        print(f"\n[3] Target Label ('{y_key}')")
        y_val = f[y_key][idx]
        
        id2label = meta.get("id2label", {})
        
        if isinstance(y_val, (int, np.integer)) or y_val.ndim == 0:
            label_str = id2label.get(int(y_val), f"Unknown ID({y_val})")
            print(f"   - Index: {y_val}")
            print(f"   - Mapped Label: {label_str}")
        else:
            indices = np.where(y_val == 1)[0]
            mapped = [id2label.get(int(i), str(i)) for i in indices]
            print(f"   - Indices: {indices}")
            print(f"   - Mapped Labels: {mapped}")

    print("="*80)

def main():
    config = load_config(CONFIG_PATH)
    data_dir = PROJECT_ROOT / Path(config.get("data_dir", "data/cache"))
    split_name = config.get("split", "train")
    tokenizer_path = config.get("tokenizer_path", "Charangan/MedBERT")
    
    h5_path = data_dir / f"train_coding_{split_name}.h5"
    
    # [추가] 결과물 저장 경로 설정
    output_txt_path = data_dir / f"inspection_report_{split_name}.txt"

    if not h5_path.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {h5_path}")
        return

    meta = load_metadata(data_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        tokenizer = None

    print(f"🚀 Analyzing {h5_path.name}...")
    print(f"📄 Report will be saved to: {output_txt_path}")

    # [수정] 파일을 열고 stdout을 리다이렉트하여 모든 print 문이 파일에 쓰이도록 함
    with open(output_txt_path, "w", encoding="utf-8") as out_f:
        with redirect_stdout(out_f):
            with h5py.File(h5_path, "r") as f:
                # 1. 구조 확인
                inspect_hdf5_structure(f)
                
                # 2. [추가] 라벨 분포 확인
                analyze_label_distribution(f, meta)
                
                # 3. 랜덤 샘플 확인
                sample_key = find_key_insensitive(f.keys(), ['input_ids', 'x_num', 'x_cat', 'x'])
                
                if sample_key:
                    total_samples = f[sample_key].shape[0]
                    # 랜덤 샘플 1개 뽑기
                    idx = random.randint(0, total_samples - 1)
                    inspect_sample_features(f, idx, meta, tokenizer)
                else:
                    print("❌ 샘플링할 메인 키(input_ids, X_num 등)를 찾을 수 없습니다.")

    print("✅ Analysis Complete! Check the txt file.")

if __name__ == "__main__":
    main()