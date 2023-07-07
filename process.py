import shutil
from pathlib import Path
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd
import sys


def read_mask(path):
    mask = cv2.imread(path, 0)
    return np.where(mask > 0, 1, 0).astype('uint8')


def cal_iou(mask1, mask2):
    add = mask1 + mask2
    union = np.where(add > 0, 1, 0)
    intersection = np.where(add > 1, 1, 0)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    return iou


def cal_center_xy(contour):
    X = contour[:, 0, 0]
    Y = contour[:, 0, 1]
    xmin, xmax = min(X), max(X)
    ymin, ymax = min(Y), max(Y)
    return int((xmin + xmax) / 2), int((ymin + ymax) / 2),


def cal_contours(mask):
    try:
        con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        _, con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return con


def match_contour_areas_to_this_area(zeros_u8, contour, this_area):
    if len(contour) == 0:
        return 0, None

    ious, areas = [], []
    for tmp in contour:
        tmp_area = cv2.drawContours(zeros_u8.copy(), [tmp], -1, 1, thickness=-1)
        iou = cal_iou(tmp_area, this_area)
        areas.append(tmp_area)
        ious.append(iou)
    max_iou = max(ious)

    if max_iou == 0:
        return 0, None
    else:
        return max_iou, areas[ious.index(max_iou)]


def get_disappear_area_from_SN(zeros_u8, sn, SN_0, SN_1, SN_2):
    sn_area = cv2.drawContours(zeros_u8.copy(), [sn], -1, 1, thickness=-1)
    sn_area_value = np.sum(sn_area)

    is_disappear_flag = False

    if len(SN_0) == 0:  # 当 Sn+1不存在时，直接把Sn判定为消失区域
        is_disappear_flag = True
    else:
        # 1 开始执行执行第一行条件
        SN_0_ious, SN_0_areas = [], []
        for tmp in SN_0:
            tmp_area = cv2.drawContours(zeros_u8.copy(), [tmp], -1, 1, thickness=-1)
            iou = cal_iou(tmp_area, sn_area)
            SN_0_areas.append(tmp_area)
            SN_0_ious.append(iou)
        SN_0_max_iou = max(SN_0_ious)

        if SN_0_max_iou == 0:  # 1.1 若 iou(Sn, Sn+1) = 0, 则Sn一定是消失区域
            is_disappear_flag = True
        else:
            sn_0_area = SN_0_areas[SN_0_ious.index(SN_0_max_iou)]
            sn_0_area_value = np.sum(sn_0_area)
            if sn_0_area_value > sn_area_value:  # 1.2 满足 iou(Sn, Sn+1) > 0 且 Sn+1 > Sn, 认为Sn也是消失区域
                is_disappear_flag = True

    if is_disappear_flag:  # 2 满足第一行条件后，开始执行第二行条件
        SN_1_ious, SN_1_areas = [], []
        for tmp in SN_1:
            tmp_area = cv2.drawContours(zeros_u8.copy(), [tmp], -1, 1, thickness=-1)
            iou = cal_iou(tmp_area, sn_area)
            SN_1_areas.append(tmp_area)
            SN_1_ious.append(iou)
        SN_1_max_iou = max(SN_1_ious)

        if SN_1_max_iou > 0:  # 2.1 满足 iou(Sn, Sn-1) > 0
            sn_1_area = SN_1_areas[SN_1_ious.index(SN_1_max_iou)]
            sn_1_area_value = np.sum(sn_1_area)

            SN_2_ious, SN_2_areas = [], []
            for tmp in SN_2:
                tmp_area = cv2.drawContours(zeros_u8.copy(), [tmp], -1, 1, thickness=-1)
                iou = cal_iou(tmp_area, sn_1_area)
                SN_2_areas.append(tmp_area)
                SN_2_ious.append(iou)
            SN_2_max_iou = max(SN_2_ious)

            if SN_2_max_iou > 0:  # 2.2 满足 iou(Sn-1, Sn-2) > 0
                sn_2_area = SN_2_areas[SN_2_ious.index(SN_2_max_iou)]
                sn_2_area_value = np.sum(sn_2_area)

                # 3 满足第二行条件后，开始执行第三行条件
                if sn_2_area_value > 1.05 * sn_1_area_value:  # 3.1 满足 Sn-2 > 1.05 * Sn-1
                    if sn_1_area_value > 1.05 * sn_area_value:  # 3.2 满足 Sn-1 > 1.05 * Sn
                        if sn_2_area_value > 2 * sn_area_value:  # 3.3 满足 Sn-2 > 2 * Sn

                            gray_value = 5 * sn_2_area_value / sn_area_value  # 所有条件都满足时，才计算 gray = 5 * Sn-2 / Sn

                            sn_area_origin = sn_area.astype('float32') * gray_value

                            x, y = cal_center_xy(sn)
                            radius = int(np.sqrt(sn_2_area_value) / np.sqrt(sn_area_value))
                            sn_area_circle = cv2.circle(zeros_u8.copy(), (x, y), radius, color=1, thickness=-1)
                            sn_area_circle = sn_area_circle.astype('float32') * gray_value

                            return x, y, sn_2_area_value, sn_1_area_value, sn_area_value, sn_area_circle, sn_area_origin
    return None


def process(data_dir, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_paths = [str(i) for i in Path(data_dir).glob('*.*') if 'out' in i.name]
    img_paths.sort()
    tmp = cv2.imread(img_paths[0], 0)
    h, w = tmp.shape
    result_origin = np.zeros((h, w), dtype='float32')
    result_circle = np.zeros((h, w), dtype='float32')
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    TABLE = []

    for N in range(2, len(img_paths) - 1):
        print('Processing SN at index [%5d/%5d]' % (N, len(img_paths)))
        SN_2 = cal_contours(read_mask(img_paths[N - 2]))  # Sn-2
        SN_1 = cal_contours(read_mask(img_paths[N - 1]))  # Sn-1
        SN = cal_contours(read_mask(img_paths[N]))  # Sn
        SN_0 = cal_contours(read_mask(img_paths[N + 1]))  # Sn+1

        img_name = Path(img_paths[N]).name.split('.jpg')[0]

        disappear_map_origin = np.zeros((h, w), dtype='float32')
        disappear_map_circle = np.zeros((h, w), dtype='float32')
        cnt = 1
        for sn in SN:
            disappear_info = get_disappear_area_from_SN(zeros_u8, sn, SN_0, SN_1, SN_2)
            if disappear_info is not None:
                x, y, sn_2_area_value, sn_1_area_value, sn_area_value, sn_area_circle, sn_area_origin = disappear_info
                disappear_map_origin = disappear_map_origin + sn_area_origin
                disappear_map_circle = disappear_map_circle + sn_area_circle
                TABLE.append([cnt, x, y, sn_2_area_value, sn_1_area_value, sn_area_value, img_name])
                cnt += 1

        result_origin = result_origin + disappear_map_origin
        result_circle = result_circle + disappear_map_circle

        save_disappear_img_path = save_dir + '/' + img_name + '_circle.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map_circle * 200, 0, 255).astype('uint8'))

        save_disappear_img_path = save_dir + '/' + img_name + '_origin.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map_origin * 200, 0, 255).astype('uint8'))

        # if N > 10: break

    plt.imshow(result_circle)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '_circle.jpg')
    plt.close()
    result_circle.tofile(save_dir + '_circle.bin')

    plt.imshow(result_origin)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '_origin.jpg')
    plt.close()
    result_origin.tofile(save_dir + '_origin.bin')
    print('Generating disappear areas done.')

    result_origin = result_origin / np.max(result_origin)
    result_origin[result_origin < 0.3] = 0
    result_origin[result_origin > 0.7] = 0
    Y, X = [], []
    for y in range(h):
        tmp = result_origin[y, :]
        if np.sum(tmp) > 0:
            Y.append(y)
    for x in range(w):
        tmp = result_origin[:, x]
        if np.sum(tmp) > 0:
            X.append(x)
    coord_origin_x, coord_origin_y = int(np.mean(X)), int(np.mean(Y))

    TABLE = np.array(TABLE)
    for i in range(TABLE.shape[0]):
        x, y = float(TABLE[i, 1]), float(TABLE[i, 2])
        x -= coord_origin_x
        y -= coord_origin_y
        TABLE[i, 1], TABLE[i, 2] = str(x), str(y)
    excel_path = save_dir + '.xlsx'
    TABLE = pd.DataFrame(TABLE)
    head = pd.DataFrame(np.array(['Idx', 'X', 'Y', 'Sn-2', 'Sn-1', 'Sn', 'img_name']).reshape(1, -1))
    TABLE = pd.concat([head, TABLE], ignore_index=True)
    TABLE.to_excel(excel_path, header=False, index=False)
    print('Saving table done.')


if __name__ == '__main__':
    data_dir = './data/5000_out'  # 分割模型的结果
    save_dir = './data/5000_disappear'  # 消失区域的可视化

    process(data_dir, save_dir)
