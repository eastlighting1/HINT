import pyarrow.parquet as pq
import os
import glob

def save_parquet_schemas_to_txt(output_txt_path, *file_paths):
    """
    가변 인자로 받은 Parquet 파일들의 스키마를 추출하여 텍스트 파일로 저장합니다.
    """
    
    # 입력된 파일 경로가 리스트나 튜플로 들어왔을 경우를 대비해 평탄화(flatten) 처리
    if len(file_paths) == 1 and isinstance(file_paths[0], (list, tuple)):
        file_paths = file_paths[0]

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Created At: {os.path.dirname(output_txt_path)}\n")
        f.write("=" * 50 + "\n\n")

        for file_path in file_paths:
            try:
                # 파일 존재 여부 확인
                if not os.path.exists(file_path):
                    f.write(f"[ERROR] File not found: {file_path}\n")
                    f.write("-" * 50 + "\n\n")
                    continue

                # 스키마 읽기 (데이터 로드 X, 메타데이터만 읽음)
                schema = pq.read_schema(file_path)
                
                # 텍스트 파일에 기록
                f.write(f"### File: {os.path.basename(file_path)}\n")
                f.write(f"Path: {file_path}\n")
                f.write(f"Columns: {len(schema.names)}\n")
                f.write("-" * 20 + " Schema Info " + "-" * 20 + "\n")
                f.write(str(schema)) # PyArrow 스키마 객체를 문자열로 변환
                f.write("\n\n" + "=" * 50 + "\n\n")
                
                print(f"Processed: {file_path}")

            except Exception as e:
                f.write(f"[ERROR] Failed to read {file_path}: {str(e)}\n")
                f.write("-" * 50 + "\n\n")
                print(f"Error processing {file_path}")

    print(f"\n[완료] 모든 스키마가 '{output_txt_path}'에 저장되었습니다.")

# --- 사용 예시 ---

# 1. 특정 파일들을 직접 지정하여 실행
# save_parquet_schemas_to_txt("schemas_output.txt", "data1.parquet", "data2.parquet")

# 2. glob을 사용하여 특정 폴더의 모든 parquet 파일 처리 (가장 일반적인 패턴)
target_files = glob.glob("./data/processed/*.parquet") # data 폴더 내 모든 parquet 파일
save_parquet_schemas_to_txt("all_schemas.txt", target_files)