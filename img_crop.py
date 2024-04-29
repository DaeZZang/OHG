import cv2

def kernel_avr(img, x, y, kernel):
    v_list = []
    x_start = x - ((kernel - 1) / 2)
    y_start = y - ((kernel - 1) / 2)
    for y_turn in range(kernel):
        for x_turn in range(kernel):
            _y = int(y_start + y_turn)
            _x = int(x_start + x_turn)
            v_list.append(img[_y, _x])
    v_avr = sum(v_list) / len(v_list)
    return v_avr

def crop_magic(img, stride = 6, draw_line = False, crop_padding = 6, kernel = 3, skip_stride = 1, thres = 0.7):
    '''
    img = 이미지
    stride = x, y를 몇 칸씩 전진하며 이미지 분석을 할 것인가?
    draw_line = 빨간색 줄 표시 옵션
    crop_padding = crop박스 크기를 키울 수 있는 옵션
    kernel = 커널 사이즈. 
    skip_stride = 이미지 코너 부분 몇 stride 만큼 무시할 것인지?
    thres = Crop할 배경색을 판단하는 기준값(높을 수록 안전하게 적게 자름)
    '''
    try:
        kernel_pad = int((kernel - 1) / 2)
        origin_height, origin_width, c = img.shape
        resize_img = cv2.resize(img, (1000, 1000))
        height, width, c = resize_img.shape

        hsv_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
        h, s, v_img = cv2.split(hsv_img)

        height -= kernel_pad
        width -= kernel_pad

        # x+
        x, y, v_list = kernel_pad, kernel_pad, []
        x = x + (stride * skip_stride)
        while True:
            now_v = kernel_avr(v_img, x, y, kernel)
            if now_v == 0:
                now_v += 0.0001
            v_list.append(now_v)

            y += stride
            if y >= height:
                v_avr = (sum(v_list) / len(v_list))

                if min(v_list) / v_avr < thres or v_avr / max(v_list) < thres:
                    x1 = x
                    break
                y, v_list = 0, []
                x += stride

        # y+
        x, y, v_list = kernel_pad, kernel_pad, []
        y = y + (stride * skip_stride)
        while True:
            now_v = kernel_avr(v_img, x, y, kernel)
            if now_v == 0:
                now_v += 0.0001
            v_list.append(now_v)

            x += stride
            if x >= width:
                v_avr = (sum(v_list) / len(v_list))

                if min(v_list) / v_avr < thres or v_avr / max(v_list) < thres:
                    y1 = y
                    break
                x, v_list = 0, []
                y += stride

        # x-
        x, y, v_list = width-1, kernel_pad, []
        x = x - (stride * skip_stride)
        while True:
            now_v = kernel_avr(v_img, x, y, kernel)
            if now_v == 0:
                now_v += 0.0001
            v_list.append(now_v)

            y += stride
            if y >= height:
                v_avr = (sum(v_list) / len(v_list))

                if min(v_list) / v_avr < thres or v_avr / max(v_list) < thres:
                    x2 = x
                    break
                y, v_list = 0, []
                x -= stride

        # y-
        x, y, v_list = kernel_pad, height-1, []
        y = y - (stride * skip_stride)
        while True:
            now_v = kernel_avr(v_img, x, y, kernel)
            if now_v == 0:
                now_v += 0.0001
            v_list.append(now_v)

            x += stride
            if x >= width:
                v_avr = (sum(v_list) / len(v_list))

                if min(v_list) / v_avr < thres or v_avr / max(v_list) < thres:
                    y2 = y
                    break
                x, v_list = 0, []
                y -= stride

        # Padding
        if x1 >= crop_padding:
            x1 -= crop_padding
        if y1 >= crop_padding:
            y1 -= crop_padding
        if x2 <= width - crop_padding:
            x2 += crop_padding
        if y2 <= height - crop_padding:
            y2 += crop_padding

        # 리사이즈된 bbox 복윈
        x1 = x1 / width * origin_width
        y1 = y1 / height * origin_height
        x2 = x2 / width * origin_width
        y2 = y2 / height * origin_height
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        bbox = [x1, y1, x2, y2]
        
        return bbox
    except:
        print('사진 crop 에러 발생. crop 무시.')
        full_bbox = [0, 0, origin_width, origin_height]
        return full_bbox
    
def cropbbox2originbbox(img_h, img_w, cropped_bbox, result_bbox):
    x_shift = cropped_bbox[0]
    y_shift = cropped_bbox[1]

    result_bbox[0] += x_shift
    result_bbox[1] += y_shift
    result_bbox[2] += x_shift
    result_bbox[3] += y_shift

    return result_bbox