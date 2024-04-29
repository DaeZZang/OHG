import iou_cal
import classifier
from utils.torch_utils import select_device
from utils.plots import Annotator
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords)
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path
import torch
from torchvision import transforms
import pandas as pd
import time
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
def run(
        source=ROOT / 'sample_images',  # file/dir/URL/glob, 0 for webcam
        # eval_flag=True,
        eval_source=ROOT / 'sample_labels',
):
    eval_flag = True
    # eval_flag = False
    weights = ROOT / 'weights_detector.pt'  # yolo model.pt path(s)
    imgsz = (1280, 1280)  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.25  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_crop = False  # save cropped prediction boxes
    visualize = False  # visualize features
    project = ROOT / 'detect'  # save results to project/name
    name = 'exp'  # save results to project/name
    exist_ok = False  # existing project/name ok, do not increment
    save_img = True  # not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load yolo model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # json 라벨 변환 리스트
    t_data = pd.read_csv(ROOT / 'class.csv', header=None)  # 클래스를 맞춰보기 위한 데이터 프레임
    t_data = t_data.iloc[:, [0, 1]]

    class_num = 0
    with open(ROOT / 'class.csv', 'r', encoding='UTF-8') as f:
        class_num = int(f.readlines()[-1].split(',')[0]) + 1
        print('분류기 class num : {}'.format(class_num))
        print('-------------------------------------------------')

    # Load mnistsimple classifier model
    mnistsimple_classifier_model = classifier.mnistsimple_Classifier_Model(
        class_num).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mnistsimple_classifier_model.load_state_dict(torch.load(ROOT / 'weights_classifier.pth'))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])

    # yolo Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # 전체 계산 변수
    ap_list = []
    seen = 0
    try_cnt = 0
    total_p, total_r, total_f1 = [], [], []

    for path, im, im0s, vid_cap, s, origin_img, x_pad, y_pad in dataset:  # path 로 파일 주소 받아오기
        try:
            start_time = time.time()

            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=visualize)

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image(배치마다 실행되므로 inference 모드에서 한번만 반복됨)

                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, example=str(names), pil=True)  # line_thickness

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    pred_bbox_list = iou_cal.tensor_to_list(det)

                    if eval_flag:  # gt 를 통하여 confusion matrix 를 통한 평가 코드
                        file_name = path.split(sep='/')[-1][:-4]
                        gt_list = iou_cal.load_gt_to_json(str(eval_source), file_name, t_data, x_pad, y_pad)

                        pred_bbox_gt_class_list, tp_list = iou_cal.iouCalc(gt_list, pred_bbox_list)
                        ap = iou_cal.ap_cal(tp_list, len(gt_list))
                        ap_list.append(ap)

                        ocrdataset = classifier.mnistsimple_Dataset(
                            im0s, pred_bbox_list, pred_bbox_gt_class_list, transforms=transform)
                        ocrloader = torch.utils.data.DataLoader(ocrdataset, batch_size=1, shuffle=False)
                        one_tp_num, pred_class_list = classifier.get_predictions(
                            mnistsimple_classifier_model, device, ocrloader, eval_flag=eval_flag)

                        tp_fp_list = []
                        for x in range(len(pred_bbox_gt_class_list)):
                            if pred_bbox_gt_class_list[x] == pred_class_list[x]:
                                tp_fp_list.append(0)  # tp : 0
                            else:
                                tp_fp_list.append(1)  # fp : 0

                        print(path.split(sep='/')[-1] + '\n')
                        one_p, one_r, one_f1 = classifier.get_f1_score(one_tp_num, len(gt_list), len(pred_bbox_list))
                        total_p.append(one_p)
                        total_r.append(one_r)
                        total_f1.append(one_f1)

                        # 결과 출력
                        print('단일 precision: {}'.format(round(one_p, 3)))
                        print('단일 recall: {}'.format(round(one_r, 3)))
                        print('단일 f-1 score: {}'.format(round(one_f1, 3)))
                        print('단일 detector mAP : {}\n'.format(round(ap, 3)))

                        classifier.get_results_image(annotator, t_data, ocrdataset,
                                                     pred_class_list, tp_fp_list, save_img, save_crop)

                        print('누적 precision: {}'.format(round(sum(total_p)/len(total_p), 3)))
                        print('누적 recall: {}'.format(round(sum(total_r)/len(total_r), 3)))
                        print('누적 f-1 score: {}'.format(round(sum(total_f1)/len(total_f1), 3)))
                        print('누적 detector mAP : {}\n'.format(round(sum(ap_list) / len(ap_list), 3)))

                    else:
                        ocrdataset = classifier.mnistsimple_Dataset(im0s, pred_bbox_list, [], transforms=transform)
                        ocrloader = torch.utils.data.DataLoader(ocrdataset, batch_size=1, shuffle=False)
                        pred_class_list = classifier.get_predictions(
                            mnistsimple_classifier_model, device, ocrloader, [])

                        # 결과 출력
                        classifier.get_results_image(annotator, t_data, ocrdataset,
                                                     pred_class_list, [], save_img, save_crop)

                # Stream results
                im0 = annotator.result()

                # 이미지 복원
                origin_img[y_pad:(y_pad + im0.shape[0]), x_pad:(x_pad + im0.shape[1])] = im0

                # Save results (image with detections)
                if save_img and dataset.mode == 'image':
                    cv2.imwrite(save_path, origin_img)

            # Print time (inference-only)
            print('소요 시간 : {} 초'.format(round(time.time() - start_time, 3)))
            print('에러 발생 횟수: {}'.format(try_cnt))
            print('-------------------------------------------------')

        except Exception as e:
            print(e)
            logging.error(f"An error occurred: {e}")
            traceback.print_exc()  # This will print the stack trace
            try_cnt += 1
            print('에러 발생 횟수: {}'.format(try_cnt))
            print('-------------------------------------------------')

    if save_img:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='sample_images', help='dir')
    parser.add_argument('--eval-source', type=str, default='sample_labels', help='eval json file dir')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
