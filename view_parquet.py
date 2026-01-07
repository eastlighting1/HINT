import pyarrow.parquet as pq
import pandas as pd
import os
import glob

def save_parquet_schemas_and_samples_to_txt(output_txt_path, *file_paths):
    """
    가변 인자로 받은 Parquet 파일들의 스키마와 
    각 컬럼별 랜덤 샘플 데이터(3개)를 추출하여 텍스트 파일로 저장합니다.
    """
    
    # 입력된 파일 경로가 리스트나 튜플로 들어왔을 경우를 대비해 평탄화(flatten) 처리
    if len(file_paths) == 1 and isinstance(file_paths[0], (list, tuple)):
        file_paths = file_paths[0]

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Created At: {os.getcwd()}\n") # 경로 정보 수정
        f.write("=" * 50 + "\n\n")

        for file_path in file_paths:
            try:
                # 1. 파일 존재 여부 확인
                if not os.path.exists(file_path):
                    f.write(f"[ERROR] File not found: {file_path}\n")
                    f.write("-" * 50 + "\n\n")
                    continue

                # 2. 스키마 및 데이터 읽기
                # 메타데이터(스키마) 읽기
                schema = pq.read_schema(file_path)
                
                # 데이터 읽기 (샘플링을 위해 Pandas DataFrame으로 변환)
                # 주의: 파일이 매우 클 경우 메모리 이슈가 발생할 수 있으므로, 
                # 실무에서는 필요한 컬럼만 읽거나 배치로 읽는 것이 좋습니다.
                table = pq.read_table(file_path)
                df = table.to_pandas()

                # 3. 텍스트 파일에 기본 정보 기록
                f.write(f"### File: {os.path.basename(file_path)}\n")
                f.write(f"Path: {file_path}\n")
                f.write(f"Rows: {len(df):,}, Columns: {len(schema.names)}\n") # 행 개수 추가
                
                # 4. 스키마 정보 기록
                f.write("-" * 20 + " Schema Info " + "-" * 20 + "\n")
                f.write(str(schema)) 
                f.write("\n")

                # 5. 랜덤 샘플 데이터 기록 (컬럼별 3개)
                f.write("-" * 20 + " Random Samples (3 values) " + "-" * 20 + "\n")
                
                if not df.empty:
                    # 데이터가 3개 미만이면 전체, 이상이면 3개 랜덤 샘플링
                    sample_size = min(3, len(df))
                    sampled_df = df.sample(n=sample_size)
                    
                    for col in df.columns:
                        # 해당 컬럼의 값들을 리스트로 변환
                        values = sampled_df[col].tolist()
                        # 보기 좋게 포맷팅 (None/Null 처리 포함)
                        values_str = [str(v) for v in values]
                        f.write(f"* {col}: {values_str}\n")
                else:
                    f.write("[Empty Data] 데이터가 없는 파일입니다.\n")

                f.write("\n" + "=" * 50 + "\n\n")
                
                print(f"Processed: {file_path}")

            except Exception as e:
                f.write(f"[ERROR] Failed to read {file_path}: {str(e)}\n")
                f.write("-" * 50 + "\n\n")
                print(f"Error processing {file_path}: {e}")

    print(f"\n[완료] 스키마와 샘플 데이터가 '{output_txt_path}'에 저장되었습니다.")

# --- 사용 예시 ---

# 테스트용 더미 데이터 생성 (실행 시 주석 해제하여 테스트 가능)
# df_test = pd.DataFrame({'id': range(10), 'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], 'value': [x*10 for x in range(10)]})
# df_test.to_parquet('test_data.parquet')
# save_parquet_schemas_and_samples_to_txt("schemas_output.txt", "test_data.parquet")

# glob을 사용하여 특정 폴더의 모든 parquet 파일 처리
target_files = glob.glob("./data/processed/*.parquet") 
if target_files:
    save_parquet_schemas_and_samples_to_txt("all_schemas.txt", target_files)
else:
    print("처리할 Parquet 파일을 찾지 못했습니다.")