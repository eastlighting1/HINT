import h5py
import hdf5plugin
import numpy as np
from pathlib import Path

# ==========================================
# 설정 (데이터셋 생성 시 사용된 설정과 맞춰주세요)
# ==========================================
FILE_PATH = "data/cache/train_intervention_train.h5"  # 검증할 HDF5 파일 경로
PRED_WINDOW = 4   # 예측 윈도우 크기 (시간 단위, 예: 4시간)
GAP_TIME = 6      # 관측 시점과 예측 윈도우 사이의 간격 (예: 6시간 뒤 예측이면 6)

# 클래스 매핑 (HDF5에 저장된 값과 비교)
# 일반적인 매핑: 0: Stay Off, 1: Stay On, 2: Onset, 3: Wean (순서는 데이터 생성 로직에 따라 다를 수 있음)
# 아래 딕셔너리는 출력용 텍스트입니다. 실제 값은 0, 1, 2, 3 정수입니다.
CLASS_NAMES = {
    0: "Stay Off",
    1: "Stay On",
    2: "Onset",
    3: "Wean"
}

def derive_class_from_window(window_data):
    """
    윈도우 내 데이터를 기반으로 클래스를 결정합니다.
    window_data: 0 또는 1로 구성된 numpy array
    """
    # 윈도우가 비어있거나 유효하지 않은 경우
    if len(window_data) == 0:
        return -1
    
    start_val = window_data[0]
    end_val = window_data[-1]
    is_all_zeros = np.all(window_data == 0)
    is_all_ones = np.all(window_data == 1)
    
    # 1. Stay Off (미사용): 윈도우 내에서 값이 항상 0
    if is_all_zeros:
        return 0
    
    # 2. Stay On (유지): 윈도우 내에서 값이 항상 1
    if is_all_ones:
        return 1
        
    # 3. Onset (시작): 0에서 1로 전이
    # 정의에 따라 "0으로 시작해서 1로 끝남" 또는 "0->1 변화 포함" 등으로 해석 가능
    # 여기서는 가장 일반적인 "시작점 0, 끝점 1" 로직을 적용합니다.
    if start_val == 0 and end_val == 1:
        return 2
        
    # 4. Wean (중단/이탈): 1에서 0으로 전이
    if start_val == 1 and end_val == 0:
        return 3
        
    # 예외 케이스 (예: 0 -> 1 -> 0 등 복잡한 변화)
    # 데이터 생성 로직에 따라 이 경우를 특정 클래스(예: Stay On 또는 별도 처리)로 분류했을 수 있습니다.
    return -99 # Undefined transition

def verify_dataset(file_path):
    p = Path(file_path)
    if not p.exists():
        print(f"[오류] 파일이 존재하지 않습니다: {file_path}")
        return

    print(f"Dataset Loading: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 데이터셋 존재 여부 확인
        if 'y' not in f or 'y_vent' not in f:
            print("[오류] 'y' 또는 'y_vent' 데이터셋을 찾을 수 없습니다.")
            print(f"Keys found: {list(f.keys())}")
            return

        y_true = f['y'][:]          # 저장된 정답 라벨 (Samples, Sequence Length)
        y_vent = f['y_vent'][:]     # 원본 바이너리 상태 (Samples, Total Length)

        # 차원 확인
        print(f"Shape of y (Labels): {y_true.shape}")
        print(f"Shape of y_vent (Raw): {y_vent.shape}")

        n_samples = y_true.shape[0]
        seq_len_y = y_true.shape[1] if y_true.ndim > 1 else 1
        
        match_count = 0
        total_checks = 0
        error_distribution = {}

        # 샘플 루프
        limit = n_samples
        print(f"\nVerifying first {limit} samples with Window={PRED_WINDOW}, Gap={GAP_TIME}...")
        
        for i in range(limit):
            # 시퀀스 루프 (y의 각 타임스텝에 대해 검증)
            # y와 y_vent의 길이 차이를 고려하여 인덱싱해야 함
            # 일반적으로 y[t]는 입력 시점 t 이후의 윈도우를 예측함
            
            # y 시퀀스 길이에 맞춰 반복
            for t in range(seq_len_y):
                # y_vent에서 해당 윈도우 추출
                # [가정] y 시퀀스는 입력 시퀀스와 매칭됨.
                # y[t]가 예측하는 구간은 t + GAP_TIME 부터 t + GAP_TIME + PRED_WINDOW 까지
                
                w_start = t + GAP_TIME
                w_end = w_start + PRED_WINDOW
                
                if w_end > y_vent.shape[1]:
                    continue # 범위 초과 시 스킵

                window = y_vent[i, w_start:w_end]
                
                # 저장된 라벨
                saved_label = y_true[i, t] if y_true.ndim > 1 else y_true[i]
                
                # -100 등 마스킹된 라벨은 건너뜀
                if saved_label == -100:
                    continue

                # 계산된 라벨
                derived_label = derive_class_from_window(window)
                
                # 비교
                if saved_label == derived_label:
                    match_count += 1
                else:
                    # 불일치 분석을 위한 기록
                    key = (int(saved_label), int(derived_label))
                    error_distribution[key] = error_distribution.get(key, 0) + 1
                    
                    # 디버깅용 상세 출력 (첫 5개 오류만)
                    if len(error_distribution) <= 5 and error_distribution[key] == 1:
                        print(f"\n[Mismatch Sample {i}, Time {t}]")
                        print(f"  Window Index: {w_start}~{w_end}")
                        print(f"  Raw Window: {window}")
                        print(f"  Derived Class: {derived_label} ({CLASS_NAMES.get(derived_label, 'Unknown')})")
                        print(f"  Stored Label : {saved_label} ({CLASS_NAMES.get(saved_label, 'Unknown')})")

                total_checks += 1

        # 결과 출력
        accuracy = (match_count / total_checks * 100) if total_checks > 0 else 0.0
        print("\n" + "="*40)
        print(f"Verification Results")
        print("="*40)
        print(f"Total Checks  : {total_checks}")
        print(f"Matches       : {match_count}")
        print(f"Accuracy      : {accuracy:.2f}%")
        
        if accuracy < 100.0:
            print("\n[Mismatch Distribution (Stored -> Derived)]")
            for (stored, derived), count in sorted(error_distribution.items(), key=lambda x: x[1], reverse=True):
                s_name = CLASS_NAMES.get(stored, str(stored))
                d_name = CLASS_NAMES.get(derived, str(derived))
                print(f"  {s_name} -> {d_name}: {count} cases")
            
            print("\n[Tip] 정확도가 낮다면 PRED_WINDOW(윈도우 크기)나 GAP_TIME(예측 시점) 설정을 확인해보세요.")
            print("      데이터 생성 시 0->1->0 과 같은 복잡한 패턴을 어떻게 처리했는지도 확인이 필요합니다.")

if __name__ == "__main__":
    verify_dataset(FILE_PATH)