from genericpath import isfile
import json
import os
import numpy as np
import pandas as pd

def load_gt_to_json(eval_source, file_name, t_data, x_pad, y_pad):
    gt_list = []
    
    if os.path.isfile(eval_source + '/' + file_name + '.json'):
        st_json = open(eval_source + '/' + file_name + '.json', "r", encoding='utf-8-sig')
        st_python = json.load(st_json)

        for i in range(len(st_python['Text_Coord'])):         
            one_bbox = st_python['Text_Coord'][i][0][0]
            
            x_min_cor = int(one_bbox[0]) - x_pad
            y_min_cor = int(one_bbox[1]) - y_pad
            
            x_max_cor = x_min_cor + int(one_bbox[2])
            y_max_cor = y_min_cor + int(one_bbox[3])
            
            t_data_exist = t_data[t_data[1] == st_python['Text_Coord'][i][0][1]]
            
            if t_data_exist.empty:
                text_class = -1
            else:
                text_class = pd.to_numeric(t_data_exist.iloc[0][0])
                    
            gt_list.append([int(x_min_cor), int(y_min_cor), int(x_max_cor), int(y_max_cor), text_class, st_python['Text_Coord'][i][0][1]])
    
    return gt_list

# yolo 의 output 인 tensor 자료형을 dictionary 자료형으로 변환
def tensor_to_list(results):
    
    pred_list = []
      
    for i in range(len(results)) :
        obj_list = [
            int(results[i][0].item()), # xmin
            int(results[i][1].item()), # ymin
            int(results[i][2].item()), # xmax
            int(results[i][3].item()), # ymax
            results[i][4].item() # conf
        ]
        pred_list.append(obj_list)
                        
    return pred_list

def iouCalc(gt_list, pred_list): # 2개의 bbox를 인자로 넣어서 iou를 return한다.
    
    return_class_list = [] 
    return_tp_list = []     
    used_idx_list = [] 
    
    for i in range(len(pred_list)):
        
        pred_bbox_area = (pred_list[i][2] - pred_list[i][0]) * (pred_list[i][3] - pred_list[i][1])

        temp_idx_list = []
        temp_iou_list = []

        for j in range(len(gt_list)):
            x1 = max(gt_list[j][0], pred_list[i][0])
            y1 = max(gt_list[j][1], pred_list[i][1])
            x2 = min(gt_list[j][2], pred_list[i][2])
            y2 = min(gt_list[j][3], pred_list[i][3])
            
            gt_bbox_area = (gt_list[j][2] - gt_list[j][0]) * (gt_list[j][3] - gt_list[j][1])

            overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

            iou = overlap_area / (pred_bbox_area + gt_bbox_area - overlap_area)  

            if iou > 0.5: # IoU threshold 기준
                temp_idx_list.append(j) # gt 의 index
                temp_iou_list.append(iou)

        if len(temp_idx_list):
            # 최적의 iou index 선택(하나만 택하는 기능)
            j_index = temp_idx_list[np.argmax(np.array(temp_iou_list))]
            
            if not j_index in used_idx_list:
                # 사용한 idx 기록
                used_idx_list.append(j_index)
                
                # temp_list 에서 제일 큰 IoU 값을 가지는 개체 선택
                return_class_list.append(gt_list[j_index][4]) # gt 의 class                
                return_tp_list.append([pred_list[i][4], 'tp'])
            else:
                return_tp_list.append([pred_list[i][4], 'fp'])
                return_class_list.append(-1)
            
        else:
            # detector 의 예측이 잘못된 경우
            return_tp_list.append([pred_list[i][4], 'fp'])
            return_class_list.append(-1)

    return_tp_list = [i[1] for i in sorted(return_tp_list, reverse = True)]

    return return_class_list, return_tp_list

def ap_cal(cm_list, len_gt): # cm_list 를 prediction 리스트 기준으로 넘기기
    x_recall, y_pre, xy_list = 0, 0, []
    tp = 0
    fp = 0
    
    for cm in cm_list:
        if cm == 'tp':
            tp += 1
        else:
            fp += 1

        x_recall = tp / len_gt
        y_pre = tp / (tp + fp)
        xy_list.append([x_recall, y_pre])
    cnt, ap = 0, []
    
    while not cnt == len(xy_list) - 1:
        if xy_list[cnt][0] != xy_list[cnt + 1][0]:
            rec = xy_list[cnt + 1][1] * (xy_list[cnt + 1][0] - xy_list[cnt][0])
            ap.append(rec)
        cnt += 1
    return sum(ap)