# 원하는 폴더로 이동
cd valid/labels 

# 폴더 내 모든 JSON 파일에 대해 파이썬 스크립트를 실행
for file in *.json
do
    python3 ../../jsonDetect.py "$file"
    echo "Processed $file"
done

echo "All JSON files have been processed."
