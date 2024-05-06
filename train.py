train_img_path = './crop_dataset/image'
valid_img_path = './crop_dataset/valid'
csv_path = './csv'

import torch
import os

torch.cuda.empty_cache()
# class 개수 측정
print('Class 개수 자동 설정 중...')
class_no = os.listdir(train_img_path)
class_no = len(class_no)
print('Class 개수: {}개'.format(class_no))

if torch.cuda.is_available() == True:
    device = 'cuda:2'
    print('현재 가상환경 GPU 사용 가능상태')
# else:
#     device = 'cpu'
#     print('GPU 사용 불가능 상태')

# 모델 초기화
def init_model():
    plt.rc('font', size = 10)
    global net, loss_fn, optim
    net = simple_cnn(class_no = class_no).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(net.parameters(), lr=0.001)
    
# epoch 카운터 초기화
def init_epoch():
    global epoch_cnt
    epoch_cnt = 0

def init_log():
    plt.rc('font', size = 10)
    # 모든 Log를 초기화
    global log_stack, iter_log, tloss_log, tacc_log, vloss_log, vacc_log, time_log
    iter_log, tloss_log, tacc_log, vloss_log, vacc_log = [], [], [], [], []
    time_log, log_stack = [], []

import gc
from torch.cuda import memory_allocated, empty_cache
def clear_memory():
    if device != 'cpu':
        empty_cache()
    gc.collect()
    
# 학습 알고리즘
import numpy as np
def epoch(data_loader, mode = 'train'):
    global epoch_cnt
    
    # 사용되는 변수 초기화
    iter_loss, iter_acc, last_grad_performed  = [], [], False
    
    # 1 iteration 학습 알고리즘(for문을 나오면 1 epoch 완료)
    for _data, _label in data_loader:
        data, label = _data.to(device), _label.to(device)
        
        # 1. Feed-forward
        if mode == 'train':
            net.train()
        else:
            # 학습때만 쓰이는 Dropout, Batch Mormalization을 미사용
            net.eval()

        result, _ = net(data) # 1 Batch에 대한 결과가 모든 Class에 대한 확률값으로
        _, out = torch.max(result, 1) # result에서 최대 확률값을 기준으로 예측 class 도출
        
        # 2. Loss 계산
        loss = loss_fn(result, label) # GT 와 Label 비교하여 Loss 산정
        iter_loss.append(loss.item()) # 학습 추이를 위하여 Loss를 기록
        
        # 3. 역전파 학습 후 Gradient Descent
        if mode == 'train':
            optim.zero_grad() # 미분을 통해 얻은 기울기 초기화 for 다음 epoch
            loss.backward() # 역전파 학습
            optim.step() # Gradient Descent 수행
            last_grad_performed = True # for문 나가면 epoch 카운터 += 1
            
        # 4. 정확도 계산
        acc_partial = (out == label).float().sum() # GT == Label 인 개수
        acc_partial = acc_partial / len(label) # ( TP / (TP + TN)) 해서 정확도 산출
        iter_acc.append(acc_partial.item()) # 학습 추이를 위하여 Acc. 기록

    # 역전파 학습 후 Epoch 카운터 += 1
    if last_grad_performed:
        epoch_cnt += 1
    
    clear_memory()
    
    # loss와 acc의 평균값 for 학습추이 그래프
    return np.average(iter_loss), np.average(iter_acc)

def epoch_not_finished():
    # 에폭이 끝남을 알림
    return epoch_cnt < maximum_epoch

def record_train_log(_tloss, _tacc, _time):
    # Train Log 기록용
    time_log.append(_time)
    tloss_log.append(_tloss)
    tacc_log.append(_tacc)
    iter_log.append(epoch_cnt)
    
def record_valid_log(_vloss, _vacc):
    # Validation Log 기록용
    vloss_log.append(_vloss)
    vacc_log.append(_vacc)

def last(log_list):
    # 리스트 안의 마지막 숫자를 반환(print_log 함수에서 사용)
    if len(log_list) > 0:
        return log_list[len(log_list) - 1]
    else:
        return -1

import matplotlib.pyplot as plt
def print_log(display = False):
    # 학습 추이 출력

    # 소숫점 3자리 수까지 조절
    train_loss = round(float(last(tloss_log)), 3)
    train_acc = round(float(last(tacc_log)), 3)
    val_loss = round(float(last(vloss_log)), 3)
    val_acc = round(float(last(vacc_log)), 3)
    time_spent = round(float(last(time_log)), 3)
    
    log_str = 'Epoch: {:3} | T_Loss {:5} | T_acc {:5} | V_Loss {:5} | V_acc. {:5} | \
🕒 {:5}'.format(last(iter_log), train_loss, train_acc, val_loss, val_acc, time_spent)
    
    log_stack.append(log_str) # 프린트 준비
    
    if display == True:
        # 학습 추이 그래프 출력
        hist_fig, loss_axis = plt.subplots(figsize=(10, 3), dpi=99) # 그래프 사이즈 설정
        hist_fig.patch.set_facecolor('white') # 그래프 배경색 설정

        # Loss Line 구성
        loss_t_line = plt.plot(iter_log, tloss_log, label='Train Loss', color='red', marker='o')
        loss_v_line = plt.plot(iter_log, vloss_log, label='Valid Loss', color='blue', marker='s')
        loss_axis.set_xlabel('epoch')
        loss_axis.set_ylabel('loss')

        # Acc. Line 구성
        acc_axis = loss_axis.twinx()
        acc_t_line = acc_axis.plot(iter_log, tacc_log, label='Train Acc.', color='red', marker='+')
        acc_v_line = acc_axis.plot(iter_log, vacc_log, label='Valid Acc.', color='blue', marker='x')
        acc_axis.set_ylabel('accuracy')

        # 그래프 출력
        hist_lines = loss_t_line + loss_v_line + acc_t_line + acc_v_line # 위에서 선언한 plt정보들 통합
        loss_axis.legend(hist_lines, [l.get_label() for l in hist_lines], loc = 'upper right') # 순서대로 그려주기
        loss_axis.grid() # 격자 설정
        plt.title('Learning history until epoch {}'.format(last(iter_log)))
        plt.draw()
    
    # 텍스트 로그 출력
    plt.show()
    with open('./train_log.txt', 'a', encoding = 'utf-8-sig') as f:
        f.write('{}\n'.format(log_stack[-1]))
        print(log_stack[-1])
    print('')

from tqdm import tqdm      
def run_model(net, data_loader, loss_fn, optim, scheduler, mode = 'train'):
    global epoch_cnt
    iter_loss, iter_acc, last_grad_performed = [], [], False
    
    iter_cnt = 0
    timer_start = time.time()
    for _data, _label in tqdm(data_loader):
        data, label = _data.to(device), _label.to(device)
        
        # 1. Feed-forward
        if mode == 'train':
            net.train()
            grad_mode = True
        else:
            # 학습때만 쓰이는 Dropout, Batch Mormalization을 미사용
            net.eval()
            grad_mode = False
        
        with torch.set_grad_enabled(grad_mode):
            result = net(data)
             
            # Feed Forward
            _, out = torch.max(result, 1) # result에서 최대 확률값을 기준으로 예측 class 도출
        
            # 2. Loss 계산
            loss = loss_fn(result, label) # GT 와 Label 비교하여 Loss 산정
            iter_loss.append(loss.item()) # 학습 추이를 위하여 Loss를 기록

            # 3. 역전파 학습 후 Gradient Descent
            if mode == 'train':
                optim.zero_grad() # 미분을 통해 얻은 기울기르 초기화 for 다음 epoch
                loss.backward() # 역전파 학습
                optim.step() # Gradient Descent 수행
                last_grad_performed = True # for문 나가면 epoch 카운터 += 1

            # 4. 정확도 계산
            acc_partial = (out == label).float().sum() # GT == Label 인 개수
            acc_partial = acc_partial / len(label) # ( TP / (TP + TN)) 해서 정확도 산출
            iter_acc.append(acc_partial.item()) # 학습 추이를 위하여 Acc. 기록
        now_per = round((int(iter_cnt * len(_label)) / (len(data_loader) * len(_label))* 100), 3)
        now_per = max(now_per, 0.0001)
        time_spent = round(time.time() - start_time, 3)
        time_left_sec = int(((time_spent / now_per) * 100) - time_spent)
        iter_cnt += 1
            
    ### 이번에 새로 배우는 scheduler
    scheduler.step() # Learning Rate 스케줄러 실행

    # 역전파 학습 후 Epoch 카운터 += 1
    if last_grad_performed:
        epoch_cnt += 1
    
    clear_memory()
    
    # loss와 acc의 평균값 for 학습추이 그래프
    return np.average(iter_loss), np.average(iter_acc)

import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Resize, Normalize
import numpy as np

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

img_size = (28)
transforms_train = transforms.Compose([ToTensor(), Resize((img_size,img_size)), Normalize(mean, std)])
transforms_val = transforms.Compose([ToTensor(), Resize((img_size,img_size)), Normalize(mean, std)])
transforms_test = transforms.Compose([ToTensor(), Resize((img_size,img_size)), Normalize(mean, std)])

# 빈 텍스트 파일 생성
with open('./train_log.txt', 'a', encoding = 'utf-8-sig') as f:
    f.write('')

import torch.nn as nn
from torch.utils.data import Dataset
import cv2
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augmentation=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['filename', 'label'])
        self.img_dir = img_dir
        self.imgs = self.img_labels["label"]
        self.transform = transform
        self.augmentation = augmentation
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = self.img_labels.iloc[idx, 0]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label = int(self.img_labels.iloc[idx, 1])
        image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label=label)["label"]
        return image, label
    def get_labels(self):
        return self.img_labels["label"]
    
import pandas as pd
train_set = CustomImageDataset('{}/train_labeled_img.csv'.format(csv_path), train_img_path, transform = transforms_train)
val_set = CustomImageDataset('{}/val_labeled_img.csv'.format(csv_path), valid_img_path, transform = transforms_val)

batch_size = 256
dataloader_train = torch.utils.data.DataLoader(train_set, batch_size = batch_size, drop_last = True, pin_memory = True)
dataloader_val = torch.utils.data.DataLoader(val_set, batch_size = batch_size, drop_last = True, pin_memory = True)

import torch.nn.functional as F
class simple_cnn(nn.Module):
    def __init__(self, class_no):
        super(simple_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 7, bias=False)    # output becomes 22x22
        self.conv1_bn = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(48, 96, 7, bias=False)   # output becomes 16x16
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 144, 7, bias=False)  # output becomes 10x10
        self.conv3_bn = nn.BatchNorm2d(144)
        self.conv4 = nn.Conv2d(144, 192, 7, bias=False) # output becomes 4x4
        self.conv4_bn = nn.BatchNorm2d(192)
        self.fc1 = nn.Linear(3072, class_no, bias=False)
        self.fc1_bn = nn.BatchNorm1d(class_no)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        flat1 = torch.flatten(conv4.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)

from torch.optim import Adam
from torch.optim import lr_scheduler

init_model()
init_epoch()
init_log()

maximum_epoch = 30
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.8)

# 모델 폴더 만들기
os.makedirs('./model_save', exist_ok = True)

# Training Iteration
import time

for cnt in range(maximum_epoch):
    epoch_cnt = cnt
    start_time = time.time()
    tloss, tacc = run_model(net, dataloader_train, loss_fn, optim, lr_scheduler, mode = 'train')
    end_time = time.time()
    time_taken = end_time - start_time
    record_train_log(tloss, tacc, time_taken)
    with torch.no_grad():
        vloss, vacc = run_model(net, dataloader_val, loss_fn, optim, lr_scheduler, mode = 'val')
        record_valid_log(vloss, vacc)
        torch.save(net.state_dict(), './model_save/best_model_{}.pth'.format(epoch_cnt))
    print_log(display = False)

print('\n Training completed!')