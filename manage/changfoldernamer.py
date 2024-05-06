import os
import shutil
import pandas as pd

# 기존 데이터셋 폴더 경로 설정
base_path = '../dataset/valid/crop'
# 새 데이터셋을 저장할 폴더 경로 설정
new_base_path = '../crop_dataset/valid'

# class3.csv 파일 불러오기
df = pd.read_csv('../csv/class3.csv')

# 처리한 파일 수 추적
file_count = 0

# 새 폴더 구조에 맞게 파일 이동 및 이름 변경
for _, row in df.iterrows():
    new_folder = os.path.join(new_base_path, str(row['ClassNumber']))
    old_folder = os.path.join(base_path, row['Character'])
    
    # 폴더 존재 여부 확인
    if os.path.exists(old_folder):
        # 새 디렉토리 생성
        os.makedirs(new_folder, exist_ok=True)
        
        # 이미지 파일 이동 및 이름 변경
        files = [f for f in os.listdir(old_folder) if f.endswith('.png')]
        for idx, file in enumerate(files):
            src = os.path.join(old_folder, file)
            dst = os.path.join(new_folder, f"{idx}.png")
            shutil.move(src, dst)
            file_count += 1
    else:
        print(f"폴더가 존재하지 않습니다: {old_folder}")

print(f"새 위치에 {file_count}개의 파일 이동 및 이름 변경이 완료되었습니다.")
