import shutil
from pathlib import Path
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def read_mask(path):
    mask = cv2.imread(path, 0)
    return np.where(mask > 0, 1, 0).astype('uint8')


def cal_iou(mask1, mask2):
    add = mask1 + mask2
    union = np.where(add > 0, 1, 0)
    intersection = np.where(add > 1, 1, 0)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def judge_disappear_contour(h, w, contour, contours, IOU_THRES):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area = cv2.drawContours(zeros_u8.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    if max_iou <= IOU_THRES:
        return True

    return False


def judge_disappear_contour_new(h, w, contour, contours, IOU_THRES):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area = cv2.drawContours(zeros_u8.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    if max_iou == 0:
        return True

    if max_iou > 0 and max_iou <= IOU_THRES:
        matched_idx = ious.index(max_iou)
        matched_area = areas[matched_idx]
        matched_area_val = np.sum(matched_area)  # Sn+1
        area_val = np.sum(area)  # Sn-2
        if area_val * 0.2 >= matched_area_val:
            return True

    return False


def get_disappear_area(h, w, contour, contours):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area = cv2.drawContours(zeros_u8.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    ret_contour = contour
    gray_value = 1

    if max_iou > 0:
        matched_idx = ious.index(max_iou)
        matched_area = areas[matched_idx]
        matched_area_val = np.sum(matched_area)  # Sn-2
        area_val = np.sum(area)  # Sn

        if area_val < matched_area_val:
            gray_value = matched_area_val / area_val

    ret_area = cv2.drawContours(zeros_u8.copy(), [ret_contour], -1, 1, thickness=-1)
    ret_area = ret_area.astype('float32')
    ret_area = ret_area * gray_value

    return ret_area


def get_disappear_area_new(h, w, contour, contours):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area = cv2.drawContours(zeros_u8.copy(), [contour], -1, 1, thickness=-1)
    ious, areas = [], []
    for con_tmp in contours:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area, area_tmp)
        areas.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    ret_contour = contour
    gray_value = 1

    if max_iou > 0:
        matched_idx = ious.index(max_iou)
        matched_area = areas[matched_idx]
        matched_area_val = np.sum(matched_area)  # Sn
        area_val = np.sum(area)  # Sn-2

        if area_val > matched_area_val:
            gray_value = area_val / matched_area_val

            if matched_area_val > 0.5 * matched_area_val:
                gray_value = 0

    ret_area = cv2.drawContours(zeros_u8.copy(), [ret_contour], -1, 1, thickness=-1)
    ret_area = ret_area.astype('float32')
    ret_area = ret_area * gray_value

    return ret_area


def process(data_dir, save_dir, WIN_SIZE, METHOD):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_paths = [str(i) for i in Path(data_dir).glob('*.*') if 'out' in i.name]
    img_paths.sort()
    tmp = cv2.imread(img_paths[0], 0)
    h, w = tmp.shape
    result = np.zeros((h, w), dtype='float32')

    for N in range(WIN_SIZE - 2, len(img_paths) - 1):
        path_cur = img_paths[N]  # Sn
        path_pre = img_paths[N - WIN_SIZE - 2]  # Sn-2
        path_nex = img_paths[N + 1]  # Sn+1

        mask_cur = read_mask(path_cur)
        mask_pre = read_mask(path_pre)
        mask_nex = read_mask(path_nex)

        con_cur, _ = cv2.findContours(mask_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        con_pre, _ = cv2.findContours(mask_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        con_nex, _ = cv2.findContours(mask_nex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        disappear_map = np.zeros((h, w), dtype='float32')

        if METHOD == 0:
            for j in range(len(con_cur)):
                if judge_disappear_contour(h, w, con_cur[j], con_nex, IOU_THRES=0):  # Sn和Sn+1比较，消失区域来自Sn
                    disappear_area = get_disappear_area(h, w, con_cur[j], con_pre)  # Sn和Sn-2之间的关系决定灰度值
                    disappear_map = disappear_map + disappear_area

        elif METHOD == 1:
            for j in range(len(con_pre)):
                if judge_disappear_contour_new(h, w, con_pre[j], con_nex, IOU_THRES=0.2):  # Sn-2和Sn+1比较，消失区域来自Sn-2
                    disappear_area = get_disappear_area_new(h, w, con_pre[j], con_cur)  # Sn-2和Sn之间的关系决定灰度值
                    disappear_map = disappear_map + disappear_area

        result = result + disappear_map
        print('sliding window at [%5d/%5d] done.' % (N, len(img_paths)))

        save_disappear_img_path = save_dir + '/' + Path(img_paths[N]).name + '.disappear.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map * 200, 0, 255).astype('uint8'))

    plt.imshow(result)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '.jpg')
    plt.close()
    print('done.')

    result.tofile(save_dir + '.bin')


if __name__ == '__main__':
    # WIN_SIZE = 4
    # METHOD = 0
    # data_dir = './data/100_out'
    # save_dir = './data/100_method0'

    WIN_SIZE = 4
    METHOD = 1
    data_dir = './data/5000_out'
    save_dir = './data/5000_method1'

    process(data_dir, save_dir, WIN_SIZE, METHOD)
