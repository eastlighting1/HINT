import torch
import json
import numpy as np
import h5py
import hdf5plugin
from pathlib import Path
from torch.utils.data import DataLoader

# [주의] 실제 프로젝트 폴더 구조에 맞춰 import 경로를 수정하세요.
# 예: from my_project.domain.vo import ICDConfig
try:
    from src.hint.domain.vo import ICDConfig
    from src.hint.domain.entities import ICDModelEntity
    from src.hint.services.training.automatic_icd_coding.evaluator import ICDEvaluator
    from src.hint.infrastructure.datasource import HDF5StreamingSource, collate_tensor_batch
    from src.hint.infrastructure.networks import get_network_class
    from src.hint.infrastructure.registry import FileSystemRegistry
    from src.hint.infrastructure.telemetry import RichTelemetryObserver
except ImportError:
    # 패키지명을 찾지 못할 경우를 대비해 상대 경로 등 조정 필요
    print("Import Error: 'src' 패키지를 찾을 수 없습니다. 프로젝트 구조에 맞춰 import 경로를 확인해주세요.")
    raise

def run_test():
    # 1. 초기화 (Config, Registry, Observer, Device)
    config = ICDConfig()  # vo.py의 기본값 사용 (필요 시 인자 전달)
    
    # artifacts 저장소 위치 (registry.py 기본값 기반)
    # 만약 main.py에서 base_dir를 다르게 설정했다면 여기서도 맞춰주세요.
    registry = FileSystemRegistry(base_dir="./artifacts") 
    
    observer = RichTelemetryObserver()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    observer.log("INFO", f"=== Start Testing Mode (Device: {device}) ===")

    # 2. 메타데이터 로드 (LabelEncoder, Ignored Indices 복원)
    # Service._prepare_data 로직과 동일하게 stats.json을 읽어야 정확한 평가 가능
    cache_dir = Path(config.data.data_cache_dir)
    stats_path = cache_dir / "stats.json"
    
    if not stats_path.exists():
        observer.log("ERROR", f"Stats file not found at {stats_path}. Cannot restore class info.")
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    if "icd_classes" not in stats:
        raise ValueError("icd_classes missing in stats.json")

    icd_classes = np.array(stats["icd_classes"])
    num_classes = len(icd_classes)
    
    # 제외할 클래스 인덱스 복원 (__MISSING__, __OTHER__ 등)
    excluded_labels = ["__MISSING__", "__OTHER__"]
    ignored_indices = [
        i for i, label in enumerate(icd_classes) 
        if label in excluded_labels
    ]
    
    observer.log("INFO", f"Loaded {num_classes} classes. Ignored indices: {ignored_indices}")

    # 3. Test Data Loader 준비
    # Config의 prefix를 이용해 test 파일 경로 추론 (예: train_coding_test.h5)
    prefix = config.data.input_h5_prefix
    test_h5_path = cache_dir / f"{prefix}_test.h5"
    
    if not test_h5_path.exists():
        observer.log("ERROR", f"Test data file not found at {test_h5_path}")
        return

    # 입력 차원(Feats, SeqLen) 확인을 위해 파일 잠시 열기
    with h5py.File(test_h5_path, "r") as f:
        shape = f["X_num"].shape
        num_feats = shape[1]
        ts_seq_len = shape[2]
        observer.log("INFO", f"Test Data Shape: Samples={shape[0]}, Feats={num_feats}, SeqLen={ts_seq_len}")

    # Test Source & Loader
    test_source = HDF5StreamingSource(test_h5_path, label_key="y")
    test_loader = DataLoader(
        test_source,
        batch_size=config.batch_size,
        shuffle=False, # 테스트는 순차적으로
        collate_fn=collate_tensor_batch,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # 4. 모델별 평가 수행
    results_summary = {}

    for model_name in ['GRU-D'] :
        observer.log("INFO", f"--- Testing Model: {model_name} ---")
        
        # 4-1. 네트워크 초기화 (Architecture)
        specific_cfg = config.model_configs.get(model_name, {})

        if model_name == 'GRU-D':
            specific_cfg['hidden_dim'] = 128
        
        # Service.py와 동일한 네트워크 인자 구성
        seq_len_for_model = ts_seq_len if ts_seq_len > 0 else config.max_length
        network_args = {
            "num_classes": num_classes,
            "input_dim": num_feats,
            "seq_len": seq_len_for_model,
            "dropout": config.dropout,
            **specific_cfg
        }

        try:
            NetworkClass = get_network_class(model_name)
            network = NetworkClass(**network_args)
            
            # 4-2. Entity 생성 및 가중치 로드
            entity = ICDModelEntity(network)
            full_model_name = f"{config.artifacts.model_name}_{model_name}" # 예: icd_model_MedBERT
            
            # 'best' 태그로 저장된 모델 로드
            try:
                state_dict = registry.load_model(full_model_name, "best", device)
                entity.load_state_dict(state_dict)
                observer.log("INFO", f"Successfully loaded 'best' weights for {full_model_name}")
            except FileNotFoundError:
                observer.log("WARNING", f"Checkpoint for {full_model_name} (best) not found. Skipping.")
                continue

            entity.to(device)

            # 4-3. 평가 (Evaluator 활용)
            evaluator = ICDEvaluator(
                config, 
                entity, 
                registry, 
                observer, 
                device, 
                ignored_indices=ignored_indices
            )
            
            metrics = evaluator.evaluate(test_loader)
            
            # 4-4. 결과 로깅 및 저장
            observer.log("INFO", f"Result [{model_name}]: {metrics}")
            
            # JSON 파일로 결과 저장 (artifacts/metrics/test_result_{model_name}.json)
            registry.save_json(metrics, f"test_result_{model_name}.json")
            results_summary[model_name] = metrics

        except Exception as e:
            observer.log("ERROR", f"Failed to test model {model_name}: {e}")
            import traceback
            observer.log("ERROR", traceback.format_exc())

    # 5. 최종 요약 출력
    observer.log("INFO", "=== All Tests Completed ===")
    print("\n[Final Test Summary]")
    for m_name, m_metrics in results_summary.items():
        print(f"Model: {m_name:15s} | Acc: {m_metrics['accuracy']:.4f} | F1: {m_metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    run_test()