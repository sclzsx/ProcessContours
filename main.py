from pathlib import Path
import json
from pathlib import Path
from PIL import Image
import labelme
import numpy as np
import cv2
from tqdm import tqdm
import os


def get_mask_by_image_path(image_path):
    json_path = image_path[:-3] + 'json'

    image = Image.open(image_path)
    imageHeight = image.height
    imageWidth = image.width
    img_shape = (imageHeight, imageWidth)

    with open(json_path, 'r', encoding='gb18030', errors='ignore') as f:
        data = json.load(f)

    # mask, _ = labelme.utils.shape.labelme_shapes_to_label(img_shape, data['shapes'])

    label_name_to_value = {'_background_': 0, "1": 1}
    mask, _ = labelme.utils.shape.shapes_to_label(img_shape, data['shapes'], label_name_to_value)
    
    mask = np.array(mask).astype('uint8')
    mask = np.where(mask > 0, 1, 0).astype('uint8')

    return mask

def cal_iou(mask1, mask2):
    add = mask1 + mask2
    union = np.where(add > 0, 1, 0)
    intersection = np.where(add > 1, 1, 0)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def cal_result_image(data_dir, result_path):
    img_paths = [str(i) for i in Path(data_dir).glob('*.jpg')]
    img_paths.sort()

    mask_pre = get_mask_by_image_path(img_paths[0])

    zeros = np.zeros_like(mask_pre)
    
    mask_out_acc = np.zeros_like(mask_pre)
    cv2.imwrite(img_paths[0][:-4] + '_out_acc.png', mask_out_acc)

    for path_cur in tqdm(img_paths[1:]):

        mask_cur = get_mask_by_image_path(path_cur)

        con_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        con_cur, _ = cv2.findContours(mask_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask_out = zeros.copy()

        for c_pre in con_pre:
            area_pre = cv2.drawContours(zeros.copy(), [c_pre], -1, 1, thickness=-1)
            ious, areas_cur = [], []
            for c_cur in con_cur:
                area_cur = cv2.drawContours(zeros.copy(), [c_cur], -1, 1, thickness=-1)
                iou = cal_iou(area_cur, area_pre)
                areas_cur.append(area_cur)
                ious.append(iou)
            max_iou = max(ious)

            if max_iou == 0:
                mask_out += area_pre
                mask_out_acc += area_pre
            else:
                area_cur = areas_cur[ious.index(max_iou)]

                if np.sum(area_cur) < np.sum(area_pre):
                    mask_out += area_cur
                else:
                    mask_out += area_pre

        mask_out = np.where(mask_out > 0, 1, 0).astype('uint8')
        
        cv2.imwrite(path_cur[:-4] + '_out.png', mask_out * 255)

        cv2.imwrite(path_cur[:-4] + '_out_acc.png', mask_out_acc * 3)

        mask_pre = mask_out

    cv2.imwrite(result_path, mask_pre * 255)


def cal_result_image2(data_dir, result_path):
    img_paths = [str(i) for i in Path(data_dir).glob('*.jpg')]
    img_paths.sort()

    mask_pre = get_mask_by_image_path(img_paths[0])

    zeros = np.zeros_like(mask_pre)
    
    mask_disappear = np.zeros_like(mask_pre)
    cv2.imwrite(img_paths[0][:-4] + '_disappear.png', mask_disappear)

    for path_cur in tqdm(img_paths[1:]):

        mask_cur = get_mask_by_image_path(path_cur)

        con_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        con_cur, _ = cv2.findContours(mask_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask_out = zeros.copy()

        for c_pre in con_pre:
            area_pre = cv2.drawContours(zeros.copy(), [c_pre], -1, 1, thickness=-1)
            ious, areas_cur = [], []
            for c_cur in con_cur:
                area_cur = cv2.drawContours(zeros.copy(), [c_cur], -1, 1, thickness=-1)
                iou = cal_iou(area_cur, area_pre)
                areas_cur.append(area_cur)
                ious.append(iou)
            max_iou = max(ious)

            if max_iou == 0:
                mask_disappear += area_pre
            else:
                area_cur = areas_cur[ious.index(max_iou)]

                if np.sum(area_cur) < np.sum(area_pre):
                    mask_out += area_cur
                else:
                    mask_out += area_pre

        mask_out = np.where(mask_out > 0, 1, 0).astype('uint8')
        cv2.imwrite(path_cur[:-4] + '_out.png', mask_out * 255)

        cv2.imwrite(path_cur[:-4] + '_disappear.png', mask_disappear * 255)

        mask_pre = mask_out

    cv2.imwrite(result_path, mask_disappear * 255)

def read_mask(path):
    mask = cv2.imread(path, 0)
    return np.where(mask > 0, 1, 0).astype('uint8')

def compare_contour_and_return_min(zeros, contour, contours):
    area = cv2.drawContours(zeros.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)
    if max_iou == 0:
        return contour
    else:
        matched_idx = ious.index(max_iou)
        matched_area = areas[matched_idx]
        matched_contour = contours[matched_idx]

        if np.sum(matched_area) < np.sum(area):
            return matched_contour
        else:
            return contour

def judge_contour_disappear(zeros, contour, contours):
    area = cv2.drawContours(zeros.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)
    if max_iou == 0:
        return True
        

def cal_result_image3(data_dir, result_path):
    img_paths = [str(i) for i in Path(data_dir).glob('*.jpg')]
    img_paths.sort()

    zeros_u8 = np.zeros((896, 1280), dtype='uint8')

    disappear_u32 = np.zeros((896, 1280), dtype='uint32')

    for i in Path(data_dir).glob('*update.png'):
        os.remove(str(i))

    for i in Path(data_dir).glob('*disappear.png'):
        os.remove(str(i))

    for i in range(1, len(img_paths) - 1):
        path_cur = img_paths[i][:-4] + '_out.png'
        path_pre = img_paths[i-1][:-4] + '_out.png'
        path_nex = img_paths[i+1][:-4] + '_out.png'

        if os.path.exists(img_paths[i-1][:-4] + '_out_update.png'):
            path_pre = img_paths[i-1][:-4] + '_out_update.png'

        mask_cur = read_mask(path_cur)
        mask_pre = read_mask(path_pre)
        mask_nex = read_mask(path_nex)

        con_nex, _ = cv2.findContours(mask_nex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        con_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        con_cur, _ = cv2.findContours(mask_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        con_cur_update = []
        for j in range(len(con_cur)):
            contour = compare_contour_and_return_min(zeros_u8.copy(), con_cur[j], con_pre)
            con_cur_update.append(contour)
        mask_cur_update = cv2.drawContours(zeros_u8.copy(), con_cur_update, -1, 255, thickness=-1)
        kernel = np.ones((3, 3),np.uint8)  
        mask_cur_update = cv2.erode(mask_cur_update, kernel)
        cv2.imwrite(img_paths[i][:-4] + '_out_update.png', mask_cur_update)

        con_cur_disappear = []
        for j in range(len(con_cur_update)):
            if judge_contour_disappear(zeros_u8.copy(), con_cur_update[j], con_nex):
                con_cur_disappear.append(con_cur_update[j])
        print('Find {} contours disappear'.format(len(con_cur_disappear)))
        mask_cur_disappear = cv2.drawContours(zeros_u8.copy(), con_cur_disappear, -1, 1, thickness=-1)
        cv2.imwrite(img_paths[i][:-4] + '_out_disappear.png', mask_cur_disappear * 255)

        disappear_u32 = disappear_u32 + mask_cur_disappear.astype('uint32')

        print('[{}/{}] done.'.format(i + 1, len(img_paths)-1))

    disappear_u32 = Image.fromarray(disappear_u32)
    disappear_u32.save(result_path) 
    print('disappear_u32 saved to:', result_image_path)

def cal_result_matrix(data_dir):
    img_paths = [str(i) for i in Path(data_dir).glob('*.jpg')]
    img_paths.sort()
    for img_path in tqdm(img_paths):
        mask = get_mask_by_image_path(img_path)


if __name__ == '__main__':
    data_dir = './1000'
    result_image_path = './disappear1000.png'

    cal_result_image3(data_dir, result_image_path)

    disappear_u32 = Image.open(result_image_path)
    disappear_u32 = np.array(disappear_u32)
    disappear_u8 = (((disappear_u32 - np.min(disappear_u32)) / (np.max(disappear_u32) - np.min(disappear_u32))) * 255).astype('uint8')
    cv2.imwrite(result_image_path[:-4] + '_u8.jpg', disappear_u8)

