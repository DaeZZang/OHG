import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema import EMA  # 가중치의 지수 이동 평균을 계산하기 위한 클래스
from torch.utils.data import Dataset
import cv2
from utils.plots import colors  # 사용자 정의 색상 구성 요소

class mnistsimple_Dataset(Dataset):
    def __init__(self, image, detect_bbox_list, class_list, transforms=None):
        # 초기화 함수: 이미지, 바운딩 박스 목록, 클래스 목록 및 변환을 받습니다.
        self.transforms = transforms
        self.data = []
        self.labels = []
        self.bbox_coord = []
        
        # 바운딩 박스 리스트를 순회하면서 각 객체를 추출하고 사이즈 조정
        for i in range(len(detect_bbox_list)):
            xmin, ymin, xmax, ymax = detect_bbox_list[i]
            
            # 이미지에서 객체를 잘라내고 28x28 크기로 리사이징
            self.data.append(cv2.resize(image[ymin:ymax, xmin:xmax], dsize=(28, 28), interpolation=cv2.INTER_CUBIC))
            
            # 클래스 목록이 있다면 해당 클래스를, 없다면 -1을 레이블로 사용
            self.labels.append(class_list[i] if class_list else -1)
            
            # 바운딩 박스 좌표 저장
            self.bbox_coord.append(detect_bbox_list[i][:4])
                
    def __getitem__(self, idx):
        # 데이터셋의 idx번째 아이템을 반환
        img = self.data[idx]
        label = self.labels[idx]
        
        # 변환 적용 가능 여부 확인
        if self.transforms:
            img = self.transforms(img)
        # 이미지와 레이블을 텐서로 변환
        img = torch.tensor(np.array(img))
        label = torch.tensor(np.array(int(label))) 
        
        return img, label
    
    def __len__(self):
        # 데이터셋의 길이 반환
        return len(self.data)
    
class mnistsimple_Classifier_Model(nn.Module):
    def __init__(self, class_num):
        super(mnistsimple_Classifier_Model, self).__init__()
        # 모델의 계층을 정의
        self.conv1 = nn.Conv2d(3, 48, 7, bias=False)    # 첫 번째 합성곱 계층
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # 두 번째 합성곱 계층
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # 세 번째 합성곱 계층
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False) # 네 번째 합성곱 계층
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, class_num, bias=False)  # 완전 연결 계층
        self.fc1_bn = nn.BatchNorm1d(class_num)
        
    def get_logits(self, x):
        # 입력 이미지 x에 대해 로짓을 계산하는 함수
        x = (x - 0.5) * 2.0  # 입력 정규화
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    
    def forward(self, x):
        # 모델의 예측을 수행하는 기본 메소드
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)

    
def get_predictions(model, device, iterator, eval_flag=False):
    """
    모델을 사용하여 데이터에 대한 예측을 수행합니다.
    :param model: 예측에 사용할 모델
    :param device: 모델이 실행될 디바이스 (예: 'cuda' 또는 'cpu')
    :param iterator: 데이터 로더
    :param eval_flag: True일 경우 정답과의 비교를 통해 True Positive 수를 반환합니다.
    :return: 예측된 클래스 목록, 필요한 경우 True Positive 수도 반환
    """
    model.eval()  # 모델을 평가 모드로 설정
    ema = EMA(model, decay=0.999)  # 지수 이동 평균을 사용하여 가중치를 안정화
    ema.assign(model)  # 모델에 EMA 가중치 적용

    tp_num = 0  # True Positive의 수
    return_pred = []  # 예측 결과를 저장할 리스트

    with torch.no_grad():  # 기울기 계산을 비활성화하여 메모리 사용량과 계산 시간 절약
        if eval_flag:
            # 평가 모드: 정답과 비교하여 True Positive를 계산
            for data, target in iterator:
                data = data.to(device)  # 데이터를 적절한 디바이스로 이동
                output = model(data)  # 모델에 데이터를 통과시켜 예측 수행
                pred = output.argmax(dim=1, keepdim=True)[0].item()  # 가장 확률이 높은 클래스 선택
                return_pred.append(pred)
                if pred == target.item():
                    tp_num += 1
            return tp_num, return_pred
        else:
            # 단순 예측 모드: True Positive 계산 없이 예측만 수행
            for data, target in iterator:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)[0].item()
                return_pred.append(pred)
            return return_pred

def get_f1_score(tp_num, gt_len, pred_len):
    """
    F1 스코어를 계산합니다.
    :param tp_num: True Positive의 수
    :param gt_len: 실제 Ground Truth의 수
    :param pred_len: 예측된 데이터의 수
    :return: 정밀도(precision), 재현율(recall), F1 스코어
    """
    if tp_num == 0:
        return 0, 0, 0
    
    p = tp_num / pred_len
    r = tp_num / gt_len
    
    f1 = 2 * p * r / (p + r)
    return p, r, f1

def get_results_image(annotator, t_data, ocrdataset, pred_class_list, tp_fp_list, save_img, save_crop):
        """
    결과 이미지를 얻어 어노테이션을 추가합니다.
    :param annotator: 이미지에 어노테이션을 추가하는 객체
    :param t_data: 전체 데이터셋
    :param ocrdataset: OCR 데이터셋 객체
    :param pred_class_list: 예측된 클래스 목록
    :param tp_fp_list: True Positive와 False Positive의 목록
    :param save_img: 이미지 저장 여부
    :param save_crop: 크롭된 이미지 저장 여부
    """
    for i in range(len(pred_class_list)):
        if save_img or save_crop:   # 이미지나 크롭을 저장해야 할 경우

            class_str = ''
            if pred_class_list[i] == -1:
                class_str = 'none'
            else:
                class_str = t_data[1][pred_class_list[i]]

            if len(tp_fp_list) == 0:
                annotator.box_label(ocrdataset.bbox_coord[i], class_str, color=colors(0, True))
            elif tp_fp_list[i] == 0: # tp_fp_list가 비어있거나 tp값이 0인 경우
                annotator.box_label(ocrdataset.bbox_coord[i], class_str, color=colors(0, True))  # 바운딩 박스와 클래스 레이블을 이미지에 추가
            
