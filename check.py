import os

# 디렉토리 경로 설정
dataset_dir = './dataset/train/images2'
labels_dir = './dataset/train/labels'

# 각 디렉토리의 파일 수를 세기
dataset_files_count = len([name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))])
labels_files_count = len([name for name in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, name))])

# 파일 수 비교
if dataset_files_count == labels_files_count:
    print("두 디렉토리의 파일 수가 같습니다.")
else:
    print("두 디렉토리의 파일 수가 다릅니다.")
    print(f"이미지 디렉토리의 파일 수: {dataset_files_count}")
    print(f"라벨 디렉토리의 파일 수: {labels_files_count}")
