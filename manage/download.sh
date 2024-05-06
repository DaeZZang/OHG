#!/bin/bash

# 시작 파일 키와 마지막 파일 키 설정
start=435252
end=435270 # 435994

# start부터 end까지 반복하며 aihubshell 명령어 실행
for (( filekey=$start; filekey<=$end; filekey++ ))
do
    aihubshell -mode d -datasetkey 71295 -filekey $filekey
done