import os
import pandas as pd

# 데이터셋의 기본 경로 설정
base_path = '../crop_dataset/valid'

# class3.csv 파일 불러오기
df = pd.read_csv('../csv/class3.csv')

# 이미지 파일 경로와 클래스 번호를 저장할 리스트
image_labels = []

# 각 클래스 폴더에서 이미지 파일 찾기
for _, row in df.iterrows():
    class_number = row['ClassNumber']
    folder_path = os.path.join(base_path, str(class_number))
    print(f"Checking folder: {folder_path}")  # 폴더 검사 확인
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        if not files:  # 파일이 없는 경우
            print(f"No PNG files in {folder_path}")
        for file in files:
            image_path = os.path.join(folder_path, file)
            image_labels.append([image_path, class_number])
    else:
        print(f"Folder does not exist: {folder_path}")

# DataFrame 생성 및 CSV 파일로 저장
if image_labels:
    label_df = pd.DataFrame(image_labels, columns=['ImagePath', 'ClassNumber'])
    label_df.to_csv('../csv/valid_labeled_img.csv', index=False, header=False)
    print("valid_labeled_img.csv 파일이 생성되었습니다.")
else:
    print("No images found. CSV file not created.")
