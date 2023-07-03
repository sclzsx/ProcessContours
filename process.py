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


def cal_contours(mask):
    try:
        con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        _, con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return con


def judge_disappear_area_cur_next(h, w, con_cur, con_next, iou_thres, judge_factor):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area_cur = cv2.drawContours(zeros_u8.copy(), [con_cur], -1, 1, thickness=-1)
    ious, areas_next = [], []
    for con_tmp in con_next:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area_cur, area_tmp)
        areas_next.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    if max_iou == 0:  # 若不存在相交关系，则con_cur一定是消失区域
        return True

    if max_iou > 0 and max_iou <= iou_thres:  # 若存在相交关系，需进一步通过面积判定是否采用con_cur
        matched_idx = ious.index(max_iou)
        matched_area = areas_next[matched_idx]
        matched_area_val = np.sum(matched_area)
        area_val = np.sum(area_cur)
        if judge_factor * area_val >= matched_area_val:  # 满足 0.6 * Sn >= Sn+1 时，也视为消失区域，采用该con_cur
            return True

    return False


def delete_disappear_area_cur_prepre(h, w, con_cur, con_prepre, delete_factor):
    zeros_u8 = np.zeros((h, w), dtype='uint8')

    area_cur = cv2.drawContours(zeros_u8.copy(), [con_cur], -1, 1, thickness=-1)
    ious, areas_prepre = [], []
    for con_tmp in con_prepre:
        area_tmp = cv2.drawContours(zeros_u8.copy(), [con_tmp], -1, 1, thickness=-1)
        iou = cal_iou(area_cur, area_tmp)
        areas_prepre.append(area_tmp)
        ious.append(iou)
    max_iou = max(ious)

    ret_contour = con_cur
    gray_value = 1

    if max_iou > 0:
        matched_idx = ious.index(max_iou)
        matched_area = areas_prepre[matched_idx]
        matched_area_val = np.sum(matched_area)  # Sn-2
        area_val = np.sum(area_cur)  # Sn

        if matched_area_val > area_val:  # 确保满足前提Sn-2 > Sn，否则一定完全保留该消失区域，即灰度值为默认值1
            gray_value = matched_area_val / area_val

            if area_val > delete_factor * matched_area_val:  # 满足 Sn > 0.5 * Sn-2 时，删除该消失区域，即将其灰度值置0
                gray_value = 0

    ret_area = cv2.drawContours(zeros_u8.copy(), [ret_contour], -1, 1, thickness=-1)
    ret_area = ret_area.astype('float32')
    ret_area = ret_area * gray_value

    return ret_area


def process(data_dir, save_dir, INTERSECT_IOU, JUDGE_FACTOR, DELETE_FACTOR):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_paths = [str(i) for i in Path(data_dir).glob('*.*') if 'out' in i.name]
    img_paths.sort()
    tmp = cv2.imread(img_paths[0], 0)
    h, w = tmp.shape
    result = np.zeros((h, w), dtype='float32')

    for N in range(4, len(img_paths) - 1):
        if N < 15: continue

        con_prepreprepre = cal_contours(read_mask(img_paths[N - 4]))  # Sn-4
        con_preprepre = cal_contours(read_mask(img_paths[N - 3]))  # Sn-3
        con_prepre = cal_contours(read_mask(img_paths[N - 2]))  # Sn-2
        con_pre = cal_contours(read_mask(img_paths[N - 1]))  # Sn-1
        con_cur = cal_contours(read_mask(img_paths[N]))  # Sn
        con_next = cal_contours(read_mask(img_paths[N + 1]))  # Sn+1

        disappear_map = np.zeros((h, w), dtype='float32')

        for i in range(len(con_cur)):
            if judge_disappear_area_cur_next(h, w, con_cur[i], con_next, INTERSECT_IOU,
                                             JUDGE_FACTOR):  # Sn和Sn+1比较，消失区域来自Sn
                disappear_area = delete_disappear_area_cur_prepre(h, w, con_cur[i], con_prepre,
                                                                  DELETE_FACTOR)  # Sn和Sn-2之间的关系决定消失区域灰度值
                disappear_map = disappear_map + disappear_area

        result = result + disappear_map
        print('sliding window at [%5d/%5d] done.' % (N, len(img_paths)))

        save_disappear_img_path = save_dir + '/' + Path(img_paths[N]).name + '.disappear.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map * 200, 0, 255).astype('uint8'))

        if N > 25: break

    plt.imshow(result)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '.jpg')
    plt.close()
    print('done.')

    result.tofile(save_dir + '.bin')


if __name__ == '__main__':
    data_dir = './data/5000_out'  # 分割模型的结果
    save_dir = './data/5000_method1_test3'  # 消失区域的可视化
    INTERSECT_IOU = 0.2  # 判断存在相交的IOU阈值
    JUDGE_FACTOR = 0.6  # 初判消失区域的参数（步骤一）
    DELETE_FACTOR = 0.5  # 删除消失区域的参数（步骤二）
    process(data_dir, save_dir, INTERSECT_IOU, JUDGE_FACTOR, DELETE_FACTOR)
