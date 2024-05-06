import json
import sys

def convert_format(data):
    return {
        "Text_Coord": [
            [[bbox["Bbox"], bbox["annotate"]]] for bbox in data["Text_Coord"]
        ]
    }

def process_json(file_path):
    # 파일 이름에서 확장자를 제외하고 경로도 제거합니다.
    base_name = file_path.split('/')[-1].replace('.json', '')

    # JSON 파일을 열고 데이터를 로드합니다
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 입력된 데이터를 새로운 형식으로 변환합니다
    converted_data = convert_format(data)

    # 변환된 데이터를 새로운 JSON 파일로 저장합니다
    output_path = f'/home/sungjun/ohg/01.AI모델/01.AI 모델/dataset/valid/labels_detect/{base_name}.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(converted_data, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    process_json(sys.argv[1])
