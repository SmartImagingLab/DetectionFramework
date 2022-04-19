# ------------------------------------------------------------------------------------------
# The pytorch demo code for detecting the object in a specific image (fpn specific version)
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, modified by Zongxian Li, based on code from faster R-CNN
# ------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import _init_paths
from model.utils.blob import im_list_to_blob
import os
import sys
import numpy as np
from skimage.transform import resize

np.set_printoptions(suppress=True)
import argparse
import pprint
import pdb
import time
import cv2
import matplotlib.pyplot as plt
import pickle as cPickle
import torch
import math
from numpy.core.multiarray import ndarray
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# from scipy.misc import imread
from imageio import imread
from roi_data_layer.roidb_v1 import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.soft_nms.nms import soft_nms
from model.soft_nms.nms import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
# from model.fpn.fpn_cascade import _FPN
from model.fpn.resnet_IN import resnet
from astropy.io import fits
from scipy import ndimage
from photutils.aperture import CircularAperture, aperture_photometry
import pdb
import re
import warnings
import json
import joblib

warnings.filterwarnings('ignore')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='EP_detect', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res50.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res50', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="./model_save",
                        nargs=argparse.REMAINDER)  # 可改
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=1,
                        action='store_true')          # 用CUDA
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="./validation")                     # 可改
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=46, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=90, type=int)                 # 可改 fpn_161_72_2395  1-46-90
    parser.add_argument('--checksession_2', dest='checksession_2',
                        help='checksession to load model',
                        default=121, type=int)
    parser.add_argument('--checkepoch_2', dest='checkepoch_2',
                        help='checkepoch to load network',
                        default=13, type=int)
    parser.add_argument('--checkpoint_2', dest='checkpoint_2',
                        help='checkpoint to load network',
                        default=229, type=int)                   # 可改 162_26_10548  or 121-13-229
    parser.add_argument('--Model_PATH', dest='Model_PATH',
                        help='checkpoint to load network',
                        default="../Classify/result/machine_RF/RandomForest_for_ep_1.model")    # 可改 RandomForest_for_ep.model
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True,
                        action='store_true')
    # 是否训练RF分类器
    parser.add_argument('--RF_train', dest='RF_train',
                        help='train random forest',
                        default=False,
                        action='store_true')
    args = parser.parse_args()
    return args


def save_json(save_path, data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path, 'w') as file:
        json.dump(data, file)


def _get_image_blob(im, target_size):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # for target_size in :
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


# 点除的实现！
# 保证分母不为0
def divide_x_y(x, y):
    if y >= 1:
        return np.true_divide(x, y)  # 做除法
    else:
        return x  # 返回本身！


def norm(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalization
    return img


def get_xy_flux(det, img):
    '''
    修正回归框的中心点和宽高！
    :param det:
    :param img:
    :return:
    '''
    # flux_thresh = 2
    x1 = det[:, 0].astype(np.int32)
    x2 = det[:, 2].astype(np.int32)
    y1 = det[:, 1].astype(np.int32)
    y2 = det[:, 3].astype(np.int32)
    flux_pre = []
    xy_point = []
    for i in range(len(det)):
        flux_pre.append(np.max(img[y1[i]:(y2[i]+1), x1[i]:(x2[i]+1)]))   # 将原图中框内最大的像素值作为flux值！
        index = np.argmax(img[y1[i]:(y2[i]+1), x1[i]:(x2[i]+1)])
        w = x2[i] - x1[i] + 1
        h = y2[i] - y1[i] + 1
        x0 = x1[i] + index % w
        y0 = y1[i] + index // w
        xy_point.append((x0, y0, w, h))
    # # 保证分母不为0！
    # if np.min(flux_pre):
    #     _flux_pre = (1 / np.min(flux_pre)) * flux_pre * flux_thresh  # 保证_flux_pre值都大于2
    # else:
    #     raise Exception('min(flux_pre) is 0!')
    return xy_point, flux_pre


# 产生模拟图像！更合理！
def data_generate(all_xy_point_list, psf_matrix, s, flag):
    mask = np.zeros((s * 2, s * 2))  # mask作为模板，用以提取目标，做目标分割！
    psf_max = np.max(resize(psf_matrix, (s, s), anti_aliasing=True))
    for point_flux in all_xy_point_list:
        point = point_flux[:4]
        flux = point_flux[-1]
        mask_tem = np.zeros((s * 2, s * 2))  # mask作为模板，用以提取目标，做目标分割！
        scale = int(flux / psf_max)  # 计算flux值与PSF的倍数
        psf_matrix_new = psf_matrix.copy()
        if scale:
            # 放缩用的是opencv 的resize函数，采用的是线性插值
            psf_matrix_new = cv2.resize(psf_matrix_new, (s * scale, s * scale))
            psf_matrix_new = psf_matrix_new[(len(psf_matrix_new) // 2 - s // 2):(len(psf_matrix_new) // 2 + s // 2),
                             (len(psf_matrix_new) // 2 - s // 2):(len(psf_matrix_new) // 2 + s // 2)]
        psf_matrix_new = norm(cv2.resize(psf_matrix_new, (s, s)))
        # psf_matrix_new = cv.resize(psf_matrix_new, (s, s))
        # 产生模拟图像！
        # Poisson分布 ; psf_matrix_new:256*256
        x_Poisson = np.random.poisson(lam=flux, size=(1, 1))  # lam为λ size为k
        if flag:
            # 若变为1，则需要将坐标扩大一倍
            mask_tem[int(point[1] // (4096 // s)):(int(point[1] // (4096 // s)) + s),
            int(point[0] // (4096 // s)):(int(point[0] // (4096 // s)) + s)] += x_Poisson[0][0] * psf_matrix_new
        else:
            mask_tem[int(point[1] // (4096 // s)):(int(point[1] // (4096 // s)) + s),
            int(point[0] // (4096 // s)):(int(point[0] // (4096 // s)) + s)] += x_Poisson[0][0] * psf_matrix_new * \
                                                                                x_Poisson[0][0]
        # 每个中心点处的目标进行叠加
        mask += mask_tem
    # 取限定大小的模拟图像
    point_matrix_simulate = mask[s // 2:s // 2 + s, s // 2:s // 2 + s]

    return point_matrix_simulate


def cal_initial_xy(xy_p, xy_p_new, flux_list, flag):
    '''
    计算原始的中心坐标，同时去除重复检测的目标
    :param xy_p:
    :param xy_p_new:
    :param flux_list:
    :param flag: 尺度大小的标志
    :return: 更新的检测结果
    '''
    xy_p = np.array(xy_p)
    flux_np = np.array(flux_list).reshape(-1, 1)
    if flag == 256:
        xy_p *= 16
    if flag == 512:
        xy_p *= 8
    xy_f = np.concatenate((xy_p, flux_np), axis=1).tolist()
    # 事先筛除一部分重复框(挨得很近的情况)——很有必要！
    if xy_p_new:
        for i in range(len(xy_p_new)):
            xy_f = list(
                filter(lambda x: (x[0] - xy_p_new[i][0]) ** 2 + (x[1] - xy_p_new[i][1]) ** 2 > (8*2) ** 2,
                       xy_f))
        xy_p_new.extend(xy_f)
    else:
        xy_p_new.extend(xy_f)
    # xy_p_new.extend(xy_f)   # 直接添加检测结果--不做筛除！
    return xy_p_new


# calculate pression and recall——比较简洁的方法
def cal_indicators_old(save_path, det, flux_internal):
    file_name = save_path.split("/")[-1].split("_")[0]
    csv_file = "../obsinfo/obsinfo" + file_name + '.csv'
    cmos = int(save_path.split("/")[-1].split(".")[0].split("_")[-1])
    df_label = pd.read_csv(csv_file)
    df_label_1 = df_label[(df_label['cmosnum'] == cmos)]  # .index.tolist()  # coms需要转成int
    df_label_1_x = df_label_1['x'].tolist()
    df_label_1_y = df_label_1['y'].tolist()
    df_label_1_flux = df_label_1['flux/mCrab'].tolist()
    n_real = len(df_label_1_y)  # 一张图像中需要检测到的总数目

    n_se_num = len(flux_internal)  # 总的区间数
    recall_internal_num = np.zeros(n_se_num)
    err_internal_num = np.zeros(n_se_num)
    sensitivity_internal_num = np.zeros(n_se_num)
    true_num = 0
    precision_num = 0  # 提高准确度表达的一个细节！
    # 初始化指标
    offset = -7  # 检测结果的偏置！
    # 统计定位误差范围！
    # each_errr = []
    init_values = (16 * 2) ** 2
    # dic = {}
    flag = np.zeros(n_real)  # 标记已经检测到的目标位置
    file = open(save_path, 'w+')
    # if det.any():  # 若是已经检测到目标
    if det:  # 若是已经检测到目标
        precision_num = len(det)
        for t in range(precision_num):  # 优先保存小图大目标的检测结果
            min_init_values = init_values   # 保存最小的距离
            # 保存检测框的图像
            for i in range(n_real):
                y0, x0 = df_label_1_y[i], df_label_1_x[i]  # 优先算作亮源！0-4095
                radius_values = (x0 - det[t][0] + offset) ** 2 + (y0 - det[t][1] + offset) ** 2
                if radius_values < min_init_values:  # 可以调试再看-可以使用半径圆来判断！
                    if not flag[i]:  # 防止重复保存，一个目标至多被检测到一次（不会出现多个框都检测到目标而计数的情况）
                        flag[i] = 1
                        min_init_values = radius_values  # 更新最小的距离值！
                        # 另一种保存方法
                        # dic['%d:'%i] = ((xm1 + xm2) / 2, (ym1 + ym2) / 2)
                        true_num += 1
                        err = (radius_values ** 0.5) / 16
                        # each_errr.append((radius_values ** 0.5) / 16)  # 将距离最小的位置保存进来
                        file.write("location {}(x,y,flux):{:.2f} {:.2f} {:.2f}\n".format(i + 1, det[t][0], det[t][1],
                                                                                         det[t][4]))
                        if df_label_1_flux[i] >= flux_internal[-1]:
                            recall_internal_num[-1] += 1
                            err_internal_num[-1] += err
                        else:
                            for j in range(n_se_num - 1):
                                if df_label_1_flux[i] >= flux_internal[j] and df_label_1_flux[i] < flux_internal[j + 1]:
                                    recall_internal_num[j] += 1
                                    err_internal_num[j] += err
                        break  # 及时终止！一个框至多只算检测到一个目标，位置相近的目标可以用SOFT-NMS捕捉另外的框，或者调整阈值！)

        # 计算查全\查准(合并重复的)\灵敏度
    # 计算查全\查准(合并重复的)\灵敏度
    if not true_num:
        file.write("no detected object！\n")
        print("There is no detected object in the %d image of %s ！\n" % (cmos, file_name))
    for j in range(n_se_num - 1):
        sensitivity_internal_num[j] = len(list(
            filter(lambda t: t >= flux_internal[j] and t < flux_internal[j + 1],
                   df_label_1_flux)))  # 滤除小于sensitivity的数目！
    sensitivity_internal_num[-1] = len(
        list(filter(lambda t: t >= flux_internal[-1], df_label_1_flux)))  # 滤除小于sensitivity的数目！
    file.close()

    return true_num, recall_internal_num, precision_num, sensitivity_internal_num, err_internal_num

def cal_indicators_256and512(save_path, det, size):
    file_name = save_path.split("/")[-1].split("_")[0]
    csv_file = "../obsinfo/obsinfo" + file_name + '.csv'
    cmos = int(save_path.split("/")[-1].split(".")[0].split("_")[-1])
    df_label = pd.read_csv(csv_file)
    df_label_1 = df_label[(df_label['cmosnum'] == cmos)]  # .index.tolist()  # coms需要转成int
    df_label_1_x = df_label_1['x'].tolist()
    df_label_1_y = df_label_1['y'].tolist()
    n_real = len(df_label_1_y)  # 一张图像中需要检测到的总数目

    true_num = 0
    # 初始化指标
    offset = -7  # 检测结果的偏置！
    # 统计定位误差范围！
    # each_errr = []
    init_values = (16 * 2) ** 2
    # dic = {}
    flag = np.zeros(n_real)  # 标记已经检测到的目标位置
    # if det.any():  # 若是已经检测到目标
    if det:  # 若是已经检测到目标
        precision_num = len(det)
        for t in range(precision_num):  # 优先保存小图大目标的检测结果
            min_init_values = init_values   # 保存最小的距离
            # 保存检测框的图像
            for i in range(n_real):
                y0, x0 = df_label_1_y[i], df_label_1_x[i]  # 优先算作亮源！0-4095
                radius_values = (x0 - det[t][0]*(4096/size) + offset) ** 2 + (y0 - det[t][1]*(4096/size) + offset) ** 2
                if radius_values < min_init_values:  # 可以调试再看-可以使用半径圆来判断！
                    if not flag[i]:  # 防止重复保存，一个目标至多被检测到一次（不会出现多个框都检测到目标而计数的情况）
                        flag[i] = 1
                        # 另一种保存方法
                        # dic['%d:'%i] = ((xm1 + xm2) / 2, (ym1 + ym2) / 2)
                        true_num += 1
                        break  # 及时终止！一个框至多只算检测到一个目标，位置相近的目标可以用SOFT-NMS捕捉另外的框，或者调整阈值！)
        # 计算查全\查准(合并重复的)\灵敏度
    # 计算查全\查准(合并重复的)\灵敏度
    return true_num

def getAperturePhotometry(img, h_size, w_size):
    positions = [(float(w_size), float(h_size))]
    radii = 3  # 设置孔径:2.2
    aperture = CircularAperture(positions, r=radii)
    phot_table = aperture_photometry(img, aperture, method='exact')
    for col in phot_table.colnames:
        phot_table[col].info.format = '%.8g'  # for consistent table output
    result = phot_table[0][3]
    # result = str(phot_table[0]).split(' ')[-13]
    return result

def cal_indicators(save_path, det, img_mat, flux_internal, real_target_folder,false_target_folder):
    file_name = save_path.split("/")[-1].split("_")[0]
    csv_file = "../obsinfo/obsinfo" + file_name + '.csv'
    cmos =save_path.split("/")[-1].split(".")[0].split("_")[-1]
    df_label = pd.read_csv(csv_file)
    df_label_1 = df_label[(df_label['cmosnum'] == int(cmos))]  # .index.tolist()  # coms需要转成int
    df_label_1_x = df_label_1['x'].tolist()
    df_label_1_y = df_label_1['y'].tolist()
    df_label_1_flux = df_label_1['flux/mCrab'].tolist()
    n_real = len(df_label_1_y)  # 一张图像中需要检测到的总数目

    # 初始化指标
    offset = -7  # 检测结果的偏置！
    init_values = (16 * 2) ** 2  # 设定真源范围阈值
    n_se_num = len(flux_internal)  # 总的区间数
    recall_internal_num = np.zeros(n_se_num)
    sensitivity_internal_num = np.zeros(n_se_num)
    err_internal_num = np.zeros(n_se_num)
    flag = np.zeros(n_real)  # 标记已经检测到的目标位置:前面已经有过筛选，若一次检测有重复目标，则都是真源！
    true_num = 0
    precision_num = len(det)  # 提高准确度表达的一个细节！
    # 统计定位误差范围！
    # each_errr = []
    # dic = {}
    file = open(save_path, 'w+')
    # if det.any():  # 若是已经检测到目标
    if precision_num:
        x1 = det[:, 0] - det[:, 2] // 2
        x2 = det[:, 0] + det[:, 2] // 2
        y1 = det[:, 1] - det[:, 3] // 2
        y2 = det[:, 1] + det[:, 3] // 2
        for t in range(precision_num):  # 优先保存小图大目标的检测结果
            # 保存检测框的图像
            for i in range(n_real):
                if not flag[i]:  # 防止重复保存，一个目标至多被检测到一次（不会出现多个框都检测到目标而计数的情况）
                    y0, x0 = df_label_1_y[i], df_label_1_x[i]  # 优先算作亮源！0-4095
                    radius_values = (x0 - det[t][0] + offset) ** 2 + (y0 - det[t][1] + offset) ** 2
                    if radius_values < init_values:  # 可以调试再看-可以使用半径圆来判断！
                        # 另一种保存方法
                        # dic['%d:'%i] = ((xm1 + xm2) / 2, (ym1 + ym2) / 2)
                        file.write("location {}(x,y,flux):{:.2f} {:.2f} {:.2f}\n".format(i + 1, det[t][0], det[t][1],
                                                                                         det[t][4]))
                        # 统计并保存真源
                        true_num += 1
                        real_target = img_mat[max(round(y1[t] / 8), 0): min(round(y2[t] / 8) + 1, 512),
                                      max(round(x1[t] / 8), 0):min(round(x2[t] / 8) + 1, 512)]
                        real_target_file = os.path.join(real_target_folder,
                                                         'real' + '_' + file_name + '_' + cmos + '_' + str(t) + '.npy')
                        np.save(real_target_file, real_target)

                        # 统计误差
                        err = (radius_values ** 0.5) / 16
                        # each_errr.append(err)  # 将距离最小的位置保存进来
                        if df_label_1_flux[i] >= flux_internal[-1]:
                            recall_internal_num[-1] += 1
                            err_internal_num[-1] += err
                        else:
                            for j in range(n_se_num - 1):
                                if df_label_1_flux[i] >= flux_internal[j] and df_label_1_flux[i] < flux_internal[j + 1]:
                                    recall_internal_num[j] += 1
                                    err_internal_num[j] += err
                        break  # 及时终止！一个框至多只算检测到一个目标，位置相近的目标可以用SOFT-NMS捕捉另外的框，或者调整阈值！)
            else:
                # 保存假源数据
                false_target = img_mat[max(round(y1[t] / 8), 0): min(round(y2[t] / 8) + 1, 512),
                               max(round(x1[t] / 8), 0):min(round(x2[t] / 8) + 1, 512)]
                false_target_file = os.path.join(false_target_folder,
                                                 'noise' + '_' + file_name + '_' + cmos + '_' + str(t) + '.npy')
                np.save(false_target_file, false_target)

    # 计算查全\查准(合并重复的)\灵敏度
    if not true_num:
        file.write("no detected object！\n")
        print("There is no detected object in the %s image of %s ！\n" % (cmos, file_name))
    for j in range(n_se_num - 1):
        sensitivity_internal_num[j] = len(list(
            filter(lambda t: t >= flux_internal[j] and t < flux_internal[j + 1],
                   df_label_1_flux)))  # 滤除小于sensitivity的数目！
    sensitivity_internal_num[-1] = len(
        list(filter(lambda t: t >= flux_internal[-1], df_label_1_flux)))  # 滤除小于sensitivity的数目！
    # 及时关闭文件
    file.close()

    return true_num, recall_internal_num, precision_num, sensitivity_internal_num, err_internal_num


def possible_target_points(kernel_5, img_mat, x_y_w_h, x_y_w_h_f, n1):
    # all_point_x = []  # 用于保存所有检测到的目标点
    # all_point_y = []
    i_x_y_w_h = []  # 保存检测框内该点所在的十字路径上点的x值
    i_x_y_w_h_f = []  # 保存检测框内该点所在的十字路径上点的x值
    # 针对暗源的阈值
    if n1 == 512:
        min_scale = 0.20
        min_corrcoef = 0.0    # 0.0
        photometry_value_init = 8   # 8以上
    else:
        # 针对亮源的阈值——前两个阈值设置小一点，减轻作用！
        min_scale = 0.38    # 比例值达到0.4
        min_corrcoef = 0.60     # 0.8 有误检(对那种条状结果进行过滤)
        photometry_value_init = 10   # 12 ——保证精确度！
    psf_col = kernel_5.reshape(kernel_5.size, order='C')
    width = kernel_5.shape[0]

    for i in range(len(x_y_w_h)):
        x1 = max(round(x_y_w_h[i][0] - x_y_w_h[i][2] // 2), 0)
        y1 = max(round(x_y_w_h[i][1] - x_y_w_h[i][3] // 2), 0)
        x2 = min(round(x_y_w_h[i][0] + x_y_w_h[i][2] // 2 + 1), n1 - 1)
        y2 = min(round(x_y_w_h[i][1] + x_y_w_h[i][3] // 2 + 1), n1 - 1)
        img = img_mat[y1:y2, x1:x2]
        h_size = (y2 - y1) / 2
        w_size = (x2 - x1) / 2
        # 条件1
        photometry_value = getAperturePhotometry(img, h_size, w_size)
        # 条件2
        real_i_num = 0  # 计数
        i_num = 0
        corrcoef = 0
        try:
            small_m = resize(img, (width, width), preserve_range=True)  # 将检测框resize为5*5大小
            small_m_col = small_m.reshape(small_m.size, order='C')
            corrcoef = abs(np.corrcoef(small_m_col, psf_col)[0, 1])
            # 条件3
            for j in range(len(small_m)):
                for t in range(len(small_m[0])):
                    # 5*5点的总数
                    i_num += small_m[j][t]
                    # 十字路径的点计数
                    if abs(j - width//2) <= 0 or abs(t - width//2) <= 0:
                        real_i_num += small_m[j][t]  # or 1
        except ValueError:
            print(x_y_w_h[i])

        # 符合点的初步判断条件：最小比例达到min_scale
        if i_num > 0 and corrcoef >= min_corrcoef and (
                real_i_num / i_num) >= min_scale and photometry_value >= photometry_value_init:
            i_x_y_w_h.append(x_y_w_h[i])
            i_x_y_w_h_f.append(x_y_w_h_f[i])

    return i_x_y_w_h, i_x_y_w_h_f


def possible_target_points_old(image, x_y_w_h, list_flux):
    # all_point_x = []  # 用于保存所有检测到的目标点
    # all_point_y = []
    i_x_y_w_h = []  # 保存检测框内该点所在的十字路径上点的x值
    i_list_flux = []
    s = 512
    for i in range(len(x_y_w_h)):
        x1 = max(int(x_y_w_h[i][0] - x_y_w_h[i][2] / 2), 0)
        x1_w = max(int(x_y_w_h[i][0] - x_y_w_h[i][2] / 10), 0)
        y1 = max(int(x_y_w_h[i][1] - x_y_w_h[i][3] / 2), 0)
        y1_h = max(int(x_y_w_h[i][1] - x_y_w_h[i][3] / 10), 0)
        x2 = min(int(x_y_w_h[i][0] + x_y_w_h[i][2] / 2), s - 1)
        x2_w = min(int(x_y_w_h[i][0] + x_y_w_h[i][2] / 10), s - 1)
        y2 = min(int(x_y_w_h[i][1] + x_y_w_h[i][3] / 2), s - 1)
        y2_h = min(int(x_y_w_h[i][1] + x_y_w_h[i][3] / 10), s - 1)
        part_image = image[y1:y2, x1:x2]
        part_image = part_image.reshape(part_image.size, order='C').tolist()
        i_num = len(part_image) - part_image.count(0)  # 检测框不为0的数量
        part_image_w = image[y1:y2, x1_w:x2_w]
        part_image_h = image[y1_h:y2_h, x1:x2]
        part_image_wh = image[y1_h:y2_h, x1_w:x2_w]
        part_image_w = part_image_w.reshape(part_image_w.size, order='C').tolist()
        part_image_h = part_image_h.reshape(part_image_h.size, order='C').tolist()
        part_image_wh = part_image_wh.reshape(part_image_wh.size, order='C').tolist()
        real_i_num_w = len(part_image_w) - part_image_w.count(0)  # 纵向十字路径中不为0的数量
        part_i_num_h = len(part_image_h) - part_image_h.count(0)  # 横向十字路径中不为0的数量
        part_i_num_wh = len(part_image_wh) - part_image_wh.count(0)  # 目标中心处不为0的数量
        real_i_num = real_i_num_w + part_i_num_h - part_i_num_wh  # 十 字路径中不为0的数量

        min_scale = 0.70  # 最小比例设置！太小的话起不到作用，太大的话会有正确的检测目标也过滤掉
        # 符合点的初步判断条件：最小比例达到min_scale
        if i_num:
            if (real_i_num / i_num) >= min_scale:
                i_x_y_w_h.append(x_y_w_h[i])
                i_list_flux.append(list_flux[i])
    return i_x_y_w_h, i_list_flux


def use_classify_4096(det, img_fits, Model_PATH):

    target_result = []
    object_size = 13
    if len(det):  # 若是已经检测到目标
        x1 = det[:, 0] - det[:, 2] // 2
        x2 = det[:, 0] + det[:, 2] // 2
        y1 = det[:, 1] - det[:, 3] // 2
        y2 = det[:, 1] + det[:, 3] // 2
        n = len(det)
        classifier_model = joblib.load(Model_PATH)
        test_data = np.zeros((n, object_size * object_size))

        while n:
            n -= 1
            # 准备数据——4维
            data = resize(img_fits[max(round(y1[n] / 8), 0): min(round(y2[n] / 8) + 1, 512),
                          max(round(x1[n] / 8), 0):min(round(x2[n] / 8) + 1, 512)], (object_size, object_size), preserve_range=True)
            test_data[n, :] = data.reshape(data.size, order='C')

        # 模型测试
        predicted = classifier_model.predict(test_data)

        for i in range(len(predicted)):
            if predicted[i] == 0:
                target_result.append(det[i])

    return target_result


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        os.remove(os.path.join(path, i))
        # c_path = os.path.join(path, i)
        # # 删除文件夹
        # if os.path.isdir(c_path):
        #     del_file(c_path)
        # else:
        #     os.remove(c_path)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("GPU {} will be used\n".format("1"))

    args = parse_args()

    lr = args.lr
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    print('Called with args:')
    print(args)
    cfg.USE_GPU_NMS = args.cuda
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    # 设置随机数种子
    # 每次运行代码时设置相同的seed，则每次生成的随机数也相同，
    # 如果不设置seed，则每次生成的随机数都会不一样
    np.random.seed(cfg.RNG_SEED)

    # load_dir 模型目录   args.net 网络   args.dataset 数据集
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name_1 = os.path.join(input_dir,
                               'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print(load_name_1)

    # 第二轮检测！——thresh要较低！（找到更多）
    # load_dir 模型目录   args.net 网络   args.dataset 数据集
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name_2 = os.path.join(input_dir,
                               'fpn_{}_{}_{}.pth'.format(args.checksession_2, args.checkepoch_2, args.checkpoint_2))
    print(load_name_2)

    pascal_classes = np.asarray(['__background__', 'EP'])
    # initilize the network here.
    # class-agnostic 方式只回归2类bounding box，即前景和背景
    if args.net == 'res50':
        fpn1 = resnet(pascal_classes, 64, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        # 到了pdb.set_trace()那就会定下来，就可以看到调试的提示符(Pdb)了
        pdb.set_trace()

    if args.net == 'res50':
        fpn2 = resnet(pascal_classes, 64, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        # 到了pdb.set_trace()那就会定下来，就可以看到调试的提示符(Pdb)了
        pdb.set_trace()

    fpn1.create_architecture()
    fpn1.cuda()
    print("load checkpoint %s" % (load_name_1))
    if args.cuda > 0:
        checkpoint = torch.load(load_name_1)
    fpn1.load_state_dict(checkpoint['model'])

    fpn2.create_architecture()
    fpn2.cuda()
    print("load checkpoint %s" % (load_name_2))
    if args.cuda > 0:
        checkpoint = torch.load(load_name_2)
    fpn2.load_state_dict(checkpoint['model'])

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    im_data2 = torch.FloatTensor(1)
    im_info2 = torch.FloatTensor(1)
    num_boxes2 = torch.LongTensor(1)
    gt_boxes2 = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        im_data2 = im_data.cuda()
        im_info2 = im_info.cuda()
        num_boxes2 = num_boxes.cuda()
        gt_boxes2 = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    im_data2 = Variable(im_data2)
    im_info2 = Variable(im_info2)
    num_boxes2 = Variable(num_boxes2)
    gt_boxes2 = Variable(gt_boxes2)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fpn1.cuda()
        fpn2.cuda()

    # 对dropout和batch normalization的操作在训练和测试的时候是不一样的
    # pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    fpn1.eval()
    fpn2.eval()

    start = time.time()
    # max_per_image = 2500
    thresh = 0.92
    thresh2 = 0.95
    vis = True
    # Set up webcam or get image directories
    # 加载PSF信息
    psf_matrix = np.load('psf_matrix_512.npy')
    # 开始进行预检测
    kernel_21 = resize(psf_matrix, (21, 21), anti_aliasing=True)
    kernel_7 = resize(kernel_21, (7, 7), anti_aliasing=True)
    kernel_5 = resize(kernel_21, (5, 5), anti_aliasing=True)
    kernel_3 = resize(kernel_21, (3, 3), anti_aliasing=True)

    # 保存以及加载分类模型
    real_target_folder = '../Classify/ep_data/ep_real_data'
    false_target_folder ='../Classify/ep_data/ep_noise_data'
    # ## 判断文件夹是否为空
    # if not os.listdir(real_target_folder):
    #     print("文件夹为空")
    # else:
    #     print("文件夹不空，需要删除文件")
    #     del_file(real_target_folder)
    #
    # if not os.listdir(false_target_folder):
    #     print("文件夹为空")
    # else:
    #     print("文件夹不空，需要删除文件")
    #     del_file(false_target_folder)

    # 初始化指标
    flux_interval = np.append(np.linspace(0.1, 1.0, 10), np.linspace(1.25, 5, 16)) # 1，2
    # flux_interval = np.linspace(0, 10, 41)# 1，2
    all_recall = np.zeros(len(flux_interval))
    all_flux_internal_num = np.zeros(len(flux_interval))
    precision_all = 0   # 最终检测的结果数量
    all_ture_num = 0    # 最终检测正确的结果
    num_all = 0            # 检测到极亮候选目标的数量
    num_1_all_before = 0   # 网络1后处理筛除前的候选目标的数量
    num_2_all_before = 0   # 网络1后处理筛除后的候选目标的数量
    num_3_all_before = 0   # 网络2后处理筛除前的候选目标的数量
    num_1_all_after = 0   # 网络3后处理筛除后的候选目标的数量
    num_2_all_after = 0   # 网络2后处理筛除前的候选目标的数量
    num_3_all_after = 0   # 网络3后处理筛除后的候选目标的数量
    del_num_1_all = 0   # 网络2后处理筛除掉的候选目标的数量
    del_num_2_all = 0   # 网络2后处理筛除掉的候选目标的数量
    del_num_3_all = 0   # 针对合并后的结果，分类器筛除掉的候选目标的数量

    real_all = 0
    real_1_all_before = 0
    real_1_all_after = 0
    real_2_all_before = 0
    real_2_all_after = 0

    mean_err_all = np.zeros(len(flux_interval))
    # real_corr_all, err_corr_all, real_scale_all, err_scale_all, real_astr_all, err_astr_all = [], [], [], [], [], []
    # real_corr_all2, err_corr_all2, real_scale_all2, err_scale_all2, real_astr_all2, err_astr_all2 = [], [], [], [], [], []
    # 第一轮检测！——thresh要较高！

    # imglist = os.listdir(args.image_dir)
    import glob
    fits_list = os.path.join(args.image_dir, '*_1_*.fits')  # 加载第一种类型的图像  1190001
    # fits_list2 = os.path.join(args.image_dir, '*_2_*.fits')  # 加载第二种类型的图像，也可以直接将1改为2
    imglist = sorted(glob.glob(fits_list),
                     key=lambda x: int(re.findall("[0-9]+", x.split('_')[-1])[0]))  # 按照数字大小排序！
    # imglist2 = sorted(glob.glob(fits_list2), key=lambda x: int(re.findall("[0-9]+", x.split('_')[-1])[0]))
    num_images = len(imglist)
    # num_images = 500
    print('Loaded Photo: {} images.'.format(num_images))
    total_tic = time.time()
    while (num_images > 0):  # num_images >=0  ？
        box_list = list()
        box_list2 = list()
        num_images -= 1
        # Load the demo image

        # im_file = os.path.join(args.image_dir, imglist[num_images])
        # im_file2 = os.path.join(args.image_dir, imglist2[num_images])
        im_file = imglist[num_images]
        im_file2 = im_file.split('_')[0] + "_2_" + im_file.split('_')[2]  # 生成对应图像的文件名
        img1 = fits.open(im_file, ignore_missing_end=True)[0].data
        all_xy_point_list = []
        result_save_path = im_file[:-4].rstrip(".") + ".save.txt"
        # erosion = cv2.erode(point_matrix, kernel)
        # 卷积：使得图像更具具备psf特性
        # kernel = resize(kernel, (3, 3))
        filter_img = ndimage.convolve(img1, kernel_3 / np.max(kernel_3))
        _, thresh_img = cv2.threshold(filter_img, 300, 255,
                                      cv2.THRESH_BINARY)  # 图像的二值化  cv.THRESH_BINARY_INV+cv.THRESH_OTSU = 1+8=9
        dealed_img = cv2.dilate(thresh_img, kernel_21)  # 将一个源的光子簇合并为一个
        contours, hierarchy = cv2.findContours(dealed_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        xy_point = []
        s = 256
        for i, contour in enumerate(contours):
            # 删除边界点
            contour = np.reshape(contour, (-1, 2))
            n = len(contour)
            while n:
                n -= 1
                a = contour[n]
                if (0 in a) or ((s - 1) in a):
                    contour = np.delete(contour, n, 0)
            contour = np.reshape(contour, (-1, 1, 2))
            contours[i] = contour  # 重新赋值！
            x, y, w, h = cv2.boundingRect(contour)  # contour是一个轮廓点集合
            if w != 0 and h != 0:
                if x <= 0 or y <= 0:
                    print(x, y, w, h)
                elif x + w > s or y + h > s:
                    print(x, y, w, h)
                else:
                    xy_point.append((x, y, x + w, y + h))
        # 若是检测到极亮源，则将其从mask掉
        if xy_point:
            box_np_1 = np.array(xy_point)  # 按照列进行拼接！也可以是box_list
            # 搜寻亮源所在的位置
            xy_point_list, flux_list = get_xy_flux(box_np_1, img1)
            # 统计真源数量
            num_all += len(xy_point_list)
            real_all += cal_indicators_256and512(result_save_path, xy_point_list, s)

            all_xy_point_list = cal_initial_xy(xy_point_list, all_xy_point_list, flux_list, s)
            flag = 0  # 两次系数乘积
            point_matrix_simulate = data_generate(all_xy_point_list, psf_matrix, s, flag)
            # 构造掩膜
            # 满足大于0.2的值保留，不满足的设为0, 模拟的数据（去除的更彻底）
            point_matrix_simulate = np.where(point_matrix_simulate > 1, point_matrix_simulate, 0)
            point_matrix_simulate = np.where(point_matrix_simulate == 0, 1, 0)
            # img_matrix_1 = img1.reshape(img1.size, order='C').tolist()
            # point_matrix_simulate = point_matrix_simulate.reshape(point_matrix_simulate.size, order='C').tolist()
            # img_matrix_div = np.array(list(map(divide_x_y, img_matrix_1, point_matrix_simulate))).reshape((s, s))
            # den_img_1 = np.where(img_matrix_div > 1, img_matrix_div, 0)  # 满足大于1的值保留，不满足的设为0;原始图像本身都是大于1的
            img_1 = np.multiply(img1, point_matrix_simulate)  # 点乘做mask
            # img_1 = cv2.filter2D(den_img_1, -1, kernel / np.max(kernel))  # -1：保证处理之后的图像深度保持一致
            # img_1 = ndimage.convolve(img_1, kernel / np.max(kernel))
        else:
            # 卷积：使得图像更具具备psf特性
            img_1 = np.copy(img1)
            # img_1 = ndimage.convolve(img_1, kernel / np.max(kernel))

        # ### use log transpoze
        # im = np.log(1 + np.abs(img1))
        max_value = np.max(img_1)
        min_value = np.min(img_1)
        mean_value = np.mean(img_1)
        # var_value = np.var(img_1)
        im_in = (img_1 - min_value) / (max_value - min_value)
        # im_in = (img1 - mean_value)/var_value

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)  # 扩展为三维数据
        # rgb -> bgr
        # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
        # line[::-1]字符串反过来 line = "abcde" line[::-1] 结果为：'edcba'
        im = im_in[:, :, ::-1]

        target_size = cfg.TEST.SCALES[0]
        # target_size = 256
        blobs, im_scales = _get_image_blob(im, target_size)  ##图片变换 该文件上面定义的函数，返回处理后的值 和尺度
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # 将tensor的大小调整为指定的大小。
        # 如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        # print(im_data.shape)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        # print(im_info.shape)
        gt_boxes.resize_(1, 5).zero_()
        # print(gt_boxes.shape)
        num_boxes.resize_(1).zero_()
        # print(num_boxes.shape)

        # pdb.set_trace()
        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn1(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            # model.rpn.bbox_transform 根据anchor和偏移量计算proposals
            # 最后返回的是左上和右下顶点的坐标[x1,y1,x2,y2]。
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            # 将改变坐标信息后超过图像边界的框的边框裁剪一下,使之在图像边界之内
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            # #Numpy的 tile() 函数,就是将原矩阵横向、纵向地复制，这里是横向
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        # print("detect_time_1:", detect_time)

        for j in range(1, len(pascal_classes)):  # 对每个类别进行遍历画框
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)  # 返回不为0的索引
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, 0.1)  # can alter
                # keep = soft_nms(cls_dets)##　use soft_nms or nms  ——是否可以转化为cpu_nms()
                # keep = nms(cls_dets, cfg.TEST.NMS)
                ###  error : data type not understood
                # cls_dets = cls_dets([keep.view(-1).long()])
                cls_dets = keep

                if pascal_classes[j] == "EP":  # 已改
                    class_name_index = 100
                    class_name_column = [class_name_index] * cls_dets.shape[0]
                    class_name = np.array(class_name_column).reshape(len(class_name_column), 1)
                    cls_dets = np.concatenate([cls_dets], axis=1)

                box_list.append(cls_dets)
        misc_toc = time.time()
        # print("根据阈值进行画框的时间1：", misc_toc-misc_tic)

        # # 读取512*512的原图图像！——若是没有检测到亮源，直接载入第二类尺寸的图像！
        img2 = fits.open(im_file2, ignore_missing_end=True)[0].data
        img_conv = ndimage.convolve(img2, kernel_5)
        img_2 = img2.copy()
        result_path1 = im_file[:-4].rstrip(".") + "_det.txt"
        s = 256
        s1 = 512
        if box_list:
            # 消除亮源信息
            # 产生亮源的模拟图像
            box_np_2 = np.concatenate(box_list, axis=0)  # 按照列进行拼接！也可以是box_list
            # np.savetxt(result_path1, box_np_2, fmt="%.8f")  # 记录
            # 搜寻亮源所在的位置:中心点-宽和高
            point_list, box_list_flux = get_xy_flux(box_np_2, img1)

            # # 真源特征统计
            # real_corr_all, err_corr_all, real_scale_all, err_scale_all, real_astr_all, err_astr_all = real_err_statistic(result_path1,
            #     kernel_5, img1, point_list, s, real_corr_all, err_corr_all, real_scale_all,
            #     err_scale_all, real_astr_all, err_astr_all)

           # 是否做筛选？
            num_1_all_before += len(point_list)
            real_1_all_before += cal_indicators_256and512(result_save_path, point_list, s)
            point_list, box_list_flux = possible_target_points(kernel_5, img1, point_list, box_list_flux, s)
            num_1_all_after += len(point_list)
            real_1_all_after += cal_indicators_256and512(result_save_path, point_list, s)


            flag1 = 0    # 做一次系数乘积
            if point_list:
                # 合并检测到的不重复的目标
                all_xy_point_list = cal_initial_xy(point_list, all_xy_point_list, box_list_flux, s)
                point_matrix_simulate = data_generate(all_xy_point_list, psf_matrix, s1, flag1)
                # ### use log transpoze
                # im = np.log(1 + np.abs(im))
                # 构造掩膜
                # 满足大于0.2(腐蚀的更彻底一点)的值保留，不满足的设为0, 模拟的数据
                point_matrix_simulate = np.where(point_matrix_simulate > 0.5, point_matrix_simulate, 0)
                point_matrix_simulate = np.where(point_matrix_simulate == 0, 1, 0)
                img_2 = np.multiply(img_2, point_matrix_simulate)  # 点乘做mask
                # 二维卷积处理
                # img_2 = ndimage.convolve(img, kernel / np.max(kernel))
                # point_matrix_simulate = np.where(point_matrix_simulate==0, point_matrix_simulate, 255)
                # img_2 = cv2.bitwise_and(img_2, img_2,
                #                      mask=(255-point_matrix_simulate).astype(np.uint8))  # 即掩膜图像白色区域是对需要处理图像像素的保留，黑色区域是对需要处理图像像素的剔除
        # img_2 = np.log(1 + img_2)     # 可加：np.abs()
        max_value = np.max(img_2)
        min_value = np.min(img_2)
        mean_value = np.mean(img_2)
        # var_value = np.var(img_2)
        im_in = (img_2 - mean_value) / (max_value - min_value)
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)  # 扩展为三维数据
        # rgb -> bgr
        # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
        # line[::-1]字符串反过来 line = "abcde" line[::-1] 结果为：'edcba'
        im = im_in[:, :, ::-1]

        target_size_1 = cfg.TEST.SCALES[0]
        # target_size_1 = 512
        blobs, im_scales = _get_image_blob(im, target_size_1)  ##图片变换 该文件上面定义的函数，返回处理后的值 和尺度
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        # 将tensor的大小调整为指定的大小。
        # 如果元素个数比当前的内存大小大，就将底层存储大小调整为与新元素数目一致的大小。
        im_data2.resize_(im_data_pt.size()).copy_(im_data_pt)
        # print(im_data.shape)
        im_info2.resize_(im_info_pt.size()).copy_(im_info_pt)
        # print(im_info.shape)
        gt_boxes2.resize_(1, 5).zero_()
        # print(gt_boxes.shape)
        num_boxes2.resize_(1).zero_()
        # print(num_boxes.shape)

        # pdb.set_trace()
        det_tic2 = time.time()

        rois, cls_prob, bbox_pred, \
        _, _, _, _, _ = fpn2(im_data2, im_info2, gt_boxes2, num_boxes2)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                            cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            # model.rpn.bbox_transform 根据anchor和偏移量计算proposals
            # 最后返回的是左上和右下顶点的坐标[x1,y1,x2,y2]。
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            # 将改变坐标信息后超过图像边界的框的边框裁剪一下,使之在图像边界之内
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            # #Numpy的 tile() 函数,就是将原矩阵横向、纵向地复制，这里是横向
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc2 = time.time()
        detect_time2 = det_toc2 - det_tic2
        # print("detect_time_2:", detect_time2)

        for j in range(1, len(pascal_classes)):  # 对每个类别进行遍历画框
            inds = torch.nonzero(scores[:, j] > thresh2).view(-1)  # 返回不为0的索引
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, 0.1)  # can alter:  0.3
                # keep = soft_nms(cls_dets)##　use soft_nms or nms  ——是否可以转化为cpu_nms()
                # keep = nms(cls_dets, cfg.TEST.NMS)
                ###  error : data type not understood
                # cls_dets = cls_dets([keep.view(-1).long()])
                cls_dets = keep

                if pascal_classes[j] == "EP":  # 已改
                    class_name_index = 100
                    class_name_column = [class_name_index] * cls_dets.shape[0]
                    class_name = np.array(class_name_column).reshape(len(class_name_column), 1)
                    cls_dets = np.concatenate([cls_dets], axis=1)

                box_list2.append(cls_dets)
        # 综合暗源检测结果！
        result_path2 = im_file2[:-4].rstrip(".") + "_det.txt"
        if box_list2:
            box_np_3 = np.concatenate(box_list2, axis=0)  # 按照列进行拼接！
            # np.savetxt(result_path2, box_np_3, fmt="%.8f")
            point_list_2, box_list_flux_2 = get_xy_flux(box_np_3, img2)

            # # 真源特征统计
            # real_corr_all2, err_corr_all2, real_scale_all2, err_scale_all2, real_astr_all2, err_astr_all2 = real_err_statistic(
            #     result_path1,
            #     kernel_5, img2, point_list_2, s1, real_corr_all2, err_corr_all2, real_scale_all2,
            #     err_scale_all2, real_astr_all2, err_astr_all2)
            # 储存可能的目标点的信息-对检测到的结果进行筛选
            num_2_all_before += len(point_list_2)
            real_2_all_before += cal_indicators_256and512(result_save_path, point_list_2, s1)
            point_list_2, box_list_flux_2 = possible_target_points(kernel_5, img2, point_list_2, box_list_flux_2,
                                                                           s1)
            num_2_all_after += len(point_list_2)
            real_2_all_after += cal_indicators_256and512(result_save_path, point_list_2, s1)

            if point_list_2:
                all_xy_point_list = cal_initial_xy(point_list_2, all_xy_point_list, box_list_flux_2, s1)
        # 结果的尺寸都在4K

        # 综合检测结果！
        result_path_all = im_file[:-4].rstrip(".") + "_det_all.txt"
        all_box_np = np.array(all_xy_point_list)  # 按照列进行拼接！
        if len(all_box_np):
            np.savetxt(result_path_all, all_box_np, fmt="%.2f")
        # else:
        #     all_box_np = np.array([0])
        #     np.savetxt(result_path_all, all_box_np)  # 若未检测到，则写入0 或者不写入

        if args.RF_train:
            # 综合评价检测结果！+ 准备分类器的训练集    # img_conv ?
            true_num, recall_internal_num, precision_num, flux_internal_num, err_num = cal_indicators(result_save_path,
                                                                                                       all_box_np, img2,
                                                                                                       flux_interval,
                                                                                                       real_target_folder,
                                                                                                       false_target_folder)
        else:
            # 整体进行进行分类器筛选——选择变了
            num_3_all_before += len(all_box_np)
            all_box_np = use_classify_4096(all_box_np, img2, args.Model_PATH)
            num_3_all_after += len(all_box_np)
            # 综合评价检测结果！
            true_num, recall_internal_num, precision_num, flux_internal_num, err_num = cal_indicators_old(result_save_path,
                                                                                                 all_box_np,
                                                                                                 flux_interval)
        # 计算某一张图像中检测到的正确目标数量、flux每个区间中正确目标的数量、检测到的数量以及flux每个区间的本该找到的数量
        all_ture_num += true_num  # 记录检测目标实际找到的总的正确目标的数量
        all_recall += recall_internal_num  # 记录检测目标在flux每个区间中实际找到的正确目标的数量
        all_flux_internal_num += flux_internal_num  # 记录flux每个区间的本该找到的数量
        precision_all += precision_num  # 记录总的检测到的数量
        mean_err_all += err_num  # 记录总的检测到的数量
        # if num_images%10 == 0:
        #     print(all_ture_num)
        #     print(all_recall)
        #     print(all_flux_internal_num)
        #     print(precision_all)

        if vis:
            #  只要有一个值不为0则为 True
            # if all_box_np.any():
            if len(all_box_np):
                all_box_np = np.array(all_box_np)
                im2show = (np.log(img2 + 1))
                scale = 8
                im2show = vis_detections(im2show, all_box_np, scale)  # 经过NMS之后的进一步置信度阈值设置-显示用！
                # result_path = im_file[:-4] + "_det.jpg"
                # cv2.imwrite(result_path, im2show / np.max(im2show) * 255)  # 写入图片内容到本地
                result_path_plt = im_file[:-4] + "_detection.jpg"
                plt.figure()
                plt.imshow(np.log(im2show / np.max(im2show) + 9) * 255)
                plt.savefig(result_path_plt)
                plt.close()

    total_toc = time.time()
    print("总的检测时间：", total_toc - total_tic)

    # 评价指标：
    try:
        recall = all_recall / all_flux_internal_num  # 两个数组的除法！
        mean_errors = mean_err_all / all_recall  # 两个数组的除法！
    except ZeroDivisionError:
        print("含有分母为0，改用依次遍历区间！")
        recall = np.zeros(len(flux_interval))  # 初始化
        mean_errors = np.zeros(len(flux_interval))  # 初始化
        for j in range(len(flux_interval)):
            if all_flux_internal_num[j]:
                recall[j] = all_recall[j] / all_flux_internal_num[j]
            if all_recall[j]:
                mean_errors[j] = mean_err_all[j] / all_recall[j]
    precision = all_ture_num / precision_all
    # mean_errors = np.mean(mean_err_all)

    print("检测到的总的真源数量：", all_ture_num)
    print("检测到的总的候选数量：", precision_all)
    print("总的真源数量：", np.sum(all_flux_internal_num))
    print("检测到极亮候选源中真源的数量：", real_all)
    print("检测到极亮候选源的数量：", num_all)
    print("亮源检测模型后处理前的候选目标中真源的数量：", real_1_all_before)
    print("亮源检测模型后处理前的候选目标总数量：", num_1_all_before)
    print("亮源检测模型后处理后的候选目标中真源的数量：", real_1_all_after)
    print("亮源检测模型后处理后的候选目标总数量：", num_1_all_after)
    print("一般源检测模型后处理前的候选目标中真源的数量：", real_2_all_before)
    print("一般源检测模型后处理前候选目标总数量：", num_2_all_before)
    print("一般源检测模型后处理后的候选目标中真源的数量：", real_2_all_after)
    print("一般源检测模型后处理后候选目标总数量：", num_2_all_after)
    print("在合并结果上分类器筛除掉的候选目标的数量：", num_3_all_before - num_3_all_after)
    print('session{}_{}-{}_indicators: \n recall = {}\n precision = {:.2f} \n mean_errors = {}'.format(
        args.checksession, thresh, thresh2, recall, precision, mean_errors))

    # 生成json文件
    data = {'precision': precision, 'recall': recall.tolist()}
    # 保存为json文件
    save_json("./mul_step_json.json", data)


    # # 生成json文件
    # statistic_data = {'real_corr_all': real_corr_all, 'err_corr_all': err_corr_all, 'real_scale_all': real_scale_all,
    #                   'err_scale_all': err_scale_all, 'real_astr_all': real_astr_all, 'err_astr_all': err_astr_all}
    # # 保存为json文件
    # save_json("./bright_statistic_data.json", statistic_data)

    # # 生成json文件
    # statistic_data2 = {'real_corr_all': real_corr_all2, 'err_corr_all': err_corr_all2, 'real_scale_all': real_scale_all2,
    #                   'err_scale_all': err_scale_all2, 'real_astr_all': real_astr_all2, 'err_astr_all': err_astr_all2}
    # # 保存为json文件
    # save_json("./ordinary_statistic_data.json", statistic_data2)


    # 绘制曲线
    # plt.figure(figsize=(12, 12))
    # bar_width = 0.5
    bar_width = 1
    plt.bar(flux_interval, recall, bar_width, align='edge', color='r')  # align='center '
    plt.xlabel('flux_interval')
    plt.ylabel('recall')
    # plt.plot(flux_interval, recall, color="red", label="recall-img")
    # plt.legend(loc='best')
    plt.title('indicators_img')
    plt.savefig(
        './output/visiual_map_loss/session{}_{}-{}_indicators_img.jpg'
            .format(args.checksession, thresh, thresh2))
    plt.close()

    end = time.time()
    print("总的时间消耗：", end - start)
    print("completed!!!")
