import json
import sys

def process_json(file_path):
    # 파일 이름에서 확장자를 제외하고 경로도 제거합니다.
    base_name = file_path.split('/')[-1].replace('.json', '')

    # JSON 파일을 열고 데이터를 로드합니다
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 이미지의 너비와 높이를 변수에 저장합니다
    image_width = data['Image_width']
    image_height = data['Image_height']

    # 라벨 데이터를 저장할 리스트를 초기화합니다
    label_data = []

    # 각 텍스트 좌표를 순회하면서 라벨 데이터를 구성합니다
    for item in data['Text_Coord']:
        x, y, width, height = item['Bbox'][0], item['Bbox'][1], item['Bbox'][2], item['Bbox'][3]

        # 이미지 크기에 대한 비율로 좌측 상단 좌표와 너비, 높이를 계산합니다
        x_normalized = x / image_width
        y_normalized = y / image_height
        width_normalized = width / image_width
        height_normalized = height / image_height

        # 라벨 파일 형식에 맞게 데이터를 리스트에 추가합니다
        label_data.append(f"0 {x_normalized:.4f} {y_normalized:.4f} {width_normalized:.4f} {height_normalized:.4f}")

    # 라벨 데이터를 텍스트 파일로 저장합니다
    with open(f'/home/sungjun/ohg/01.AI모델/01.AI 모델/dataset/valid/texts/{base_name}.txt', 'w') as file:
        file.write('\n'.join(label_data))

if __name__ == "__main__":
    process_json(sys.argv[1])
