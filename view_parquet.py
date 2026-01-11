import pyarrow.parquet as pq
import pandas as pd
import os
import glob
import random

def save_parquet_schemas_and_samples_to_txt(output_txt_path, *file_paths):
    """
    가변 인자로 받은 Parquet 파일들의 스키마와 
    각 컬럼별 랜덤 샘플 데이터(3개)를 추출하여 텍스트 파일로 저장합니다.
    """
    
    # 입력된 파일 경로가 리스트나 튜플로 들어왔을 경우를 대비해 평탄화(flatten) 처리
    if len(file_paths) == 1 and isinstance(file_paths[0], (list, tuple)):
        file_paths = file_paths[0]

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Created At: {os.getcwd()}\n") 
        f.write("=" * 50 + "\n\n")

        for file_path in file_paths:
            try:
                # 1. 파일 존재 여부 확인
                if not os.path.exists(file_path):
                    f.write(f"[ERROR] File not found: {file_path}\n")
                    f.write("-" * 50 + "\n\n")
                    continue

                # 2. 메타데이터 및 테이블 읽기
                # 먼저 스키마와 테이블 정보를 읽습니다.
                try:
                    schema = pq.read_schema(file_path)
                    table = pq.read_table(file_path)
                except Exception as read_e:
                    f.write(f"[ERROR] Failed to read Parquet file structure: {read_e}\n")
                    f.write("-" * 50 + "\n\n")
                    continue

                # 3. 기본 정보 및 스키마 기록
                f.write(f"### File: {os.path.basename(file_path)}\n")
                f.write(f"Path: {file_path}\n")
                f.write(f"Rows: {table.num_rows:,}, Columns: {len(schema.names)}\n") 
                
                f.write("-" * 20 + " Schema Info " + "-" * 20 + "\n")
                f.write(str(schema)) 
                f.write("\n")

                # 4. 데이터 샘플링 및 기록 (호환성 개선 버전)
                f.write("-" * 20 + " Random Samples (3 values) " + "-" * 20 + "\n")
                
                try:
                    n_rows = table.num_rows
                    if n_rows > 0:
                        # 1) 랜덤 인덱스 3개 선정
                        sample_size = min(3, n_rows)
                        indices = sorted(random.sample(range(n_rows), sample_size))
                        
                        sample_data = {}
                        
                        # 2) 컬럼별로 순회하며 값을 하나씩 추출 (Column-wise Extraction)
                        # 'take' 함수 대신 인덱싱([])과 as_py()를 사용하여 string_view 등 최신 타입 에러 회피
                        for col_name in table.column_names:
                            col_array = table[col_name]
                            col_values = []
                            for idx in indices:
                                try:
                                    # PyArrow 스칼라 값을 Python 기본 타입으로 안전하게 변환
                                    val = col_array[idx].as_py()
                                    col_values.append(val)
                                except Exception:
                                    col_values.append("<Conversion Error>")
                            sample_data[col_name] = col_values
                        
                        # 3) DataFrame 생성 및 출력
                        df_sample = pd.DataFrame(sample_data)
                        
                        for col in df_sample.columns:
                            values = df_sample[col].tolist()
                            values_str = [str(v) for v in values]
                            f.write(f"* {col}: {values_str}\n")
                    else:
                        f.write("[Empty Data] 데이터가 없는 파일입니다.\n")

                except Exception as data_error:
                    f.write(f"[WARNING] Failed to extract data samples: {str(data_error)}\n")
                    f.write("(Error detail: This usually happens with complex nested types or unsupported codecs.)\n")

                f.write("\n" + "=" * 50 + "\n\n")
                print(f"Processed: {file_path}")

            except Exception as e:
                f.write(f"[ERROR] Unexpected error processing {file_path}: {str(e)}\n")
                f.write("-" * 50 + "\n\n")
                print(f"Error processing {file_path}: {e}")

    print(f"\n[완료] 스키마와 샘플 데이터가 '{output_txt_path}'에 저장되었습니다.")

# --- 사용 예시 ---
target_files = glob.glob("./data/processed/*.parquet") 
if target_files:
    save_parquet_schemas_and_samples_to_txt("all_schemas.txt", target_files)
else:
    print("처리할 Parquet 파일을 찾지 못했습니다.")