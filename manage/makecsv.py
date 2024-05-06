import os
import pandas as pd

# 데이터셋 폴더 경로 설정
base_path = '../dataset/valid/crop'
directories = os.listdir(base_path)

# 클래스 정보를 저장할 리스트
class_info = []

# 각 디렉토리(글자) 별로 파일 개수 카운트
for idx, directory in enumerate(sorted(directories)):
    folder_path = os.path.join(base_path, directory)
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    class_info.append([idx, directory, len(files)])

# DataFrame으로 변환하고 CSV 파일로 저장
df = pd.DataFrame(class_info, columns=['ClassNumber', 'Character', 'ImageCount'])
df.to_csv('../csv/class3_valid.csv', index=False)

print("CSV 파일이 생성되었습니다.")
