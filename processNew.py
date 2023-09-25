import shutil
from pathlib import Path
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import pandas as pd
import time
import sys


def read_mask(path):
    mask = cv2.imread(path, 0)
    return np.where(mask > 0, 1, 0).astype('uint8')


def cal_iou(mask1, mask2):
    add = mask1 + mask2
    union = np.where(add > 0, 1, 0)
    intersection = np.where(add > 1, 1, 0)
    
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    assert iou >= 0
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
        return 0, None, 0

    ious, areas = [], []
    for tmp in contour:
        tmp_area = cv2.drawContours(zeros_u8.copy(), [tmp], -1, 1, thickness=-1)
        iou = cal_iou(tmp_area, this_area)
        areas.append(tmp_area)
        ious.append(iou)
    max_iou = max(ious)

    if max_iou == 0:
        return 0, None, 0
    else:
        matched_area = areas[ious.index(max_iou)]#前一附图片上的对应区域
        #kernel = np.ones((3, 3), np.uint8)
        #res_dilation = cv2.dilate (this_area, kernel, interation = 10)
        #add = matched_area + res_dilation
        #matched_area = np.where(add > 1, 1, 0)##################*****************前一副图片上的对应区域（进行重新定义）
        return max_iou, matched_area, np.sum(matched_area)  # 返回最大iou，最大iou对应的区域，区域面积


def get_disappear_area_from_SN(zeros_u8, sn, SN_0, SN_1, SN_2):
    # 画出当前Sn区域
    sn_area = cv2.drawContours(zeros_u8.copy(), [sn], -1, 1, thickness=-1)
    sn_area_value = np.sum(sn_area)

    if sn_area_value > 400:  # 当Sn的面积大于300时，跳过该区域
        return None

    is_disappear_flag = False

    if len(SN_0) == 0:  # 当 Sn+1不存在时，直接把Sn判定为消失区域
        is_disappear_flag = True
    else:
        SN_0_max_iou, sn_0_area, sn_0_area_value = match_contour_areas_to_this_area(zeros_u8, SN_0, sn_area)

        if SN_0_max_iou == 0:  # 1.1 若 iou(Sn, Sn+1) = 0, 则Sn一定是消失区域
            is_disappear_flag = True
        else:
            if sn_0_area_value < 0.13 * sn_area_value:  # 1.2 满足 iou(Sn, Sn+1) > 0 且 Sn+1 < 0.3 Sn, 认为Sn也是消失区
                if sn_area_value < 200:  # Sn < 180, 认为Sn也是消失区
                    is_disappear_flag = True
            else:
                if sn_0_area_value < 0.6 * sn_area_value:  # 1.3 满足 iou(Sn, Sn+1) > 0 且 Sn+1 > 0.3 Sn,
                    if sn_area_value < 100:  # 1.3 满足 iou(Sn, Sn+1) > 0 且 Sn+1 < 100, 认为Sn也是消失区
                        is_disappear_flag = True
         
                
                    

    # 当前Sn区域是潜在消失区域时，继续
    if is_disappear_flag:

        SN_1_max_iou, sn_1_area, sn_1_area_value = match_contour_areas_to_this_area(zeros_u8, SN_1, sn_area)  # Sn-1所有区域与当前Sn区域做匹配，获得匹配到的Sn-1信息

        if SN_1_max_iou > 0:  # 2.1 满足 iou(Sn, Sn-1) > 0 时，继续

            SN_2_max_iou, sn_2_area, sn_2_area_value = match_contour_areas_to_this_area(zeros_u8, SN_2, sn_1_area)  # Sn-2所有区域与上一步获得的Sn-1区域做匹配，获得匹配到的Sn-2信息

            if SN_2_max_iou > 0:  # 2.2 满足 iou(Sn-1, Sn-2) > 0 时，继续                
                if sn_2_area_value > 1.1 * sn_1_area_value:  # 3.1 满足 Sn-2 > 1.1 * Sn-1 时，继续
                    if sn_1_area_value > 1.5 * sn_area_value:  # 3.2 满足 Sn-1 > 1.1 * Sn 时，继续
                        if sn_2_area_value > 3.5 * sn_area_value:  # 3.3 满足 Sn-2 > 3 * Sn 时，继续
                            if sn_2_area_value < 50 * sn_area_value:  # 3.3 满足 Sn-2 < 20 * Sn 时，继续

                             gray_value = 150 # 所有条件都满足时，才计算 gray = 50 * Sn-2 / Sn          * sn_2_area_value / sn_1_area_value

                             sn_area_origin = sn_2_area.astype('float32') * gray_value######********改用“sn-2”形状替换sn，以突出冲击作用强度

                             x, y = cal_center_xy(sn)  # 计算Sn的坐标
                             radius = int(0.3 * np.sqrt(sn_2_area_value))  # 计算画圆半径 radius = 0.6* sqrt(Sn-2) / sqrt(Sn)

                             sn_area_circle = cv2.circle(zeros_u8.copy(), (x, y), radius, color=1, thickness=-1)
                             
                             sn_area_circle = sn_area_circle.astype('float32') * gray_value######********改用“sn-2”的面积替换sn

                             return x, y, sn_2_area_value, sn_1_area_value, sn_area_value, sn_area_circle, sn_area_origin
    return None


def process():
    from unet import UNet
    from segment import demo

    net = UNet(n_classes=2)

    # 参数配置
    # pt_path = 'Results/v0-unet-h512-w512/min_valid_loss.pt'  # 训练好的模型路径
    # input_hw = (512, 512)  # 输入网络的尺寸，这里采用训练时的尺寸，其他尺寸也能跑通（但需要被16整除）
    test_img_dir = './data/500_ori'  # 原图路径
    data_dir = './data/500_segment'  # 分割结果保存路径
    save_dir = './data/500_process'  # 消失区域的保存路径

    do_segment = True

    if do_segment:
        # 删除已有的分割结果
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # 执行分割
        # demo(net, pt_path, input_hw, test_img_dir, data_dir)

        # 对Sn区域逐个处理(二值化-膨胀2-腐蚀7-膨胀5）
        img_paths = [i for i in Path(test_img_dir).glob('*.jpg')]
        print(len(img_paths))
        img_paths.sort()
        N = len(img_paths)
        TABLE = []
        cnt = 0
        for img_path in img_paths:
            img_name = img_path.name
            print('[{}/{}]'.format(cnt+1, N), img_name)
            img = cv2.imread(str(img_path), 0) / 255
            h_ori, w_ori = img.shape
            
            down_rate = 0.5 # 对输入图下采样率，为1时不做下采样
            img = cv2.resize(img, None, fx=down_rate, fy=down_rate)
            h, w = img.shape
            h2, w2 = h / 2, w / 2
            radius_rate = 0.9 # 对半径再衰减一个比例，为1时不衰减，即短边的一半
            r = min(h2, w2) * radius_rate
            
            for i in range(h):
                for j in range(w):
                    ii = i - h2
                    jj = j - w2
                    if ii*ii + jj*jj >= r*r:
                        img[i,j] = 0
                    else:
                        g = img[i,j]
                        g = g*g
                        binary_threshold = 0.01 #二化阈值，由于灰度值做了平方，该值较小
                        if g > binary_threshold:
                            img[i,j] = 1
                        else:
                            img[i,j] = 0
            img = img.astype('uint8')

            median_radius = 5 #中值滤波窗口大小
            img = cv2.medianBlur(img, median_radius)

            k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            k5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) 
            k7 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) 
            img = cv2.dilate(img, k3)
            img = cv2.erode(img, k7)
            img = cv2.dilate(img, k5)

            img = cv2.resize(img, (w_ori, h_ori))
            img = np.where(img > 0, 1, 0)

            pixel_sum = np.sum(img)
            TABLE.append([img_name, str(pixel_sum)])
            cnt += 1
            
            img = (img * 255).astype('uint8')
            cv2.imwrite(data_dir + '/' + img_name[:-4] + '_out.jpg', img)
        excel_path = data_dir + '_pixel_sum.xlsx'
        TABLE = pd.DataFrame(TABLE)
        head = pd.DataFrame(np.array(['name', 'sum']).reshape(1, -1))
        TABLE = pd.concat([head, TABLE], ignore_index=True)
        TABLE.to_excel(excel_path, header=False, index=False)


    # 删除已有的消失区域结果
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 文件名排序，读取长宽信息，初始化变量
    img_paths = [str(i) for i in Path(data_dir).glob('*.*') if '_out' in i.name]
    img_paths.sort()
    tmp = cv2.imread(img_paths[0], 0)
    h, w = tmp.shape
    result_origin = np.zeros((h, w), dtype='float32')
    result_circle = np.zeros((h, w), dtype='float32')
    zeros_u8 = np.zeros((h, w), dtype='uint8')
    TABLE = []

    # 逐帧提取消失区域，当前帧为Sn
    for N in range(2, len(img_paths) - 1):
        img_name = Path(img_paths[N]).name.split('_out')[0]
        print('Processing SN [%5d/%5d] %s' % (N, len(img_paths), img_name))

        # 求邻近各帧的轮廓
        SN_2 = cal_contours(read_mask(img_paths[N - 2]))  # Sn-2
        SN_1 = cal_contours(read_mask(img_paths[N - 1]))  # Sn-1
        SN = cal_contours(read_mask(img_paths[N]))  # Sn
        SN_0 = cal_contours(read_mask(img_paths[N + 1]))  # Sn+1（注意，代码采用SN_0表示）

        disappear_map_origin = np.zeros((h, w), dtype='float32')
        disappear_map_circle = np.zeros((h, w), dtype='float32')

        # 对Sn区域逐个处理
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

        # 保存该帧消失区域结果（提亮了100倍）
        save_disappear_img_path = save_dir + '/' + img_name + '_circle.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map_circle * 150, 0, 255).astype('uint8'))
        save_disappear_img_path = save_dir + '/' + img_name + '_origin.jpg'
        cv2.imwrite(save_disappear_img_path, np.clip(disappear_map_origin * 150, 0, 255).astype('uint8'))
    print('Generating disappear areas done.')

    # 可视化最终结果
    plt.imshow(result_circle)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '_circle.jpg')
    plt.close()
    plt.imshow(result_origin)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_dir + '_origin.jpg')
    plt.close()

    # 保存bin文件，可在matlab查看
    result_circle.tofile(save_dir + '_circle.bin')
    result_origin.tofile(save_dir + '_origin.bin')

    # 找最终结果的大致中心
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
    print('coord_origin_x: %d, coord_origin_y: %d' % (coord_origin_x, coord_origin_y))

    # 更新坐标，信息写入表格
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


if __name__ == '__main__':#当文件被运行时，执行以下命令
    process()

    time.sleep(60)
    os.system("shutdown /h /f")  # windows休眠（管理员运行时生效）
