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
import re
np.set_printoptions(suppress=True)
import argparse
import pprint
import pdb
import time
import cv2
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle as cPickle
import torch
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
from photutils.aperture import CircularAperture, aperture_photometry
import pdb
import warnings

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
                        nargs=argparse.REMAINDER)                        # 可改
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=1,
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="./validation")  # 可改
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=161, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=72, type=int)  #
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=2395, type=int)                    # 可改1-46-90
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)

    ###  begin to test
    parser.add_argument('--thresh', dest='thresh',
                        help='thresh to select scores',
                        default=0.95, type=float)           # 置信度阈值 can alter
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True,
                        action='store_true')
    args = parser.parse_args()
    return args


def _get_image_blob(im):
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

    for target_size in cfg.TEST.SCALES:
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

# calculate pression and recall
def cal_indicators_new(save_path, det, flux_internal):
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
    sensitivity_internal_num = np.zeros(n_se_num)
    true_num = 0
    offset = -7
    init_values = (16 * 2)**2
    # dic = {}
    flag = np.zeros(n_real)  # 标记已经检测到的目标位置
    precision_num = 0
    file = open(save_path, 'w+')
    if det.any():  # 若是已经检测到目标
        precision_num = len(det)
        for t in range(precision_num):  # 优先保存小图大目标的检测结果
            # 保存检测框的图像
            for i in range(n_real):
                if not flag[i]:  # 防止重复保存，一个目标至多被检测到一次（不会出现多个框都检测到目标而计数的情况）
                    y0, x0 = df_label_1_y[i], df_label_1_x[i]  # 优先算作亮源！0-4095
                    radius_values = (x0 - det[t][0] + offset) ** 2 + (y0 - det[t][1] + offset) ** 2
                    if radius_values < init_values:  # 可以调试再看-可以使用半径圆来判断！
                        flag[i] = 1
                        true_num += 1
                        # 另一种保存方法
                        # dic['%d:'%i] = ((xm1 + xm2) / 2, (ym1 + ym2) / 2
                        file.write("location {}(x,y,flux):{:.2f} {:.2f} {:.2f}\n".format(i + 1, det[t][0], det[t][1], det[t][4]))
                        if df_label_1_flux[i] >= flux_internal[-1]:
                            recall_internal_num[-1] += 1
                        else:
                            for j in range(n_se_num - 1):
                                if df_label_1_flux[i] >= flux_internal[j] and df_label_1_flux[i] < flux_internal[j + 1]:
                                    recall_internal_num[j] += 1
                        break  # 及时终止！一个框至多只算检测到一个目标，位置相近的目标可以用SOFT-NMS捕捉另外的框，或者调整阈值！)
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
    return true_num, recall_internal_num, precision_num, sensitivity_internal_num


def get_xy_flux(det, img):
    # flux_thresh = 2
    x1 = det[:, 0].astype(np.int32)
    x2 = det[:, 2].astype(np.int32)
    y1 = det[:, 1].astype(np.int32)
    y2 = det[:, 3].astype(np.int32)
    flux_pre = []
    xy_point = []
    for i in range(len(det)):
        flux_pre.append(np.max(img[y1[i]:y2[i], x1[i]:x2[i]]))   # 将原图中框内最大的像素值作为flux值！
        index = np.argmax(img[y1[i]:y2[i], x1[i]:x2[i]])
        w = x2[i] - x1[i]
        h = y2[i] - y1[i]
        x0 = x1[i] + index % w
        y0 = y1[i] + index // w
        xy_point.append((x0, y0, w, h))
    # # 保证分母不为0！
    # if np.min(flux_pre):
    #     _flux_pre = (1 / np.min(flux_pre)) * flux_pre * flux_thresh  # 保证_flux_pre值都大于2
    # else:
    #     raise Exception('min(flux_pre) is 0!')
    return xy_point, flux_pre

def cal_initial_xy(xy_p, xy_p_new, flux_list, flag):
    '''
    计算原始的中心坐标，同时去除重复检测的目标
    :param xy_p:
    :param xy_p_new:
    :param flux_list:
    :param flag: 尺度大小的标志
    :return: 更新的检测结果
    '''
    xy_p_np = np.array(xy_p)
    flux_np = np.array(flux_list).reshape(-1, 1)
    if flag == 256:
        xy_p_np *= 16
    if flag == 512:
        xy_p_np *= 8
    xy_f = np.concatenate((xy_p_np, flux_np), axis=1).tolist()
    if xy_p_new:
        for i in range(len(xy_p_new)):
            xy_f = list(
                filter(lambda x: (x[0] - xy_p_new[i][0]) ** 2 + (x[1] - xy_p_new[i][1]) ** 2 > (16*2) ** 2,
                       xy_f))
        xy_p_new.extend(xy_f)
    else:
        xy_p_new.extend(xy_f)
    return xy_p_new

def getAperturePhotometry(img, h_size, w_size):
    positions = [(float(w_size), float(h_size))]
    radii = 3   # 设置孔径:2.2
    aperture = CircularAperture(positions, r=radii)
    phot_table = aperture_photometry(img, aperture, method='exact')
    for col in phot_table.colnames:
        phot_table[col].info.format = '%.8g'  # for consistent table output
    result = phot_table[0][3]
    # result = str(phot_table[0]).split(' ')[-13]
    return result

def possible_target_points_old(img_mat, x_y_w_h, flux_list, min_num, min_scale):
    # all_point_x = []  # 用于保存所有检测到的目标点
    # all_point_y = []
    i_flux_list = []  # 保存检测框内该点所在的十字路径上点的x值
    i_x_y_w_h = []  # 保存检测框内该点所在的十字路径上点的x值
    n1 = 256
    bia = 2
    for i in range(len(x_y_w_h)):
        x0 = int(x_y_w_h[i][0])
        y0 = int(x_y_w_h[i][1])
        # 验证正确的点的相关数据！
        small_m = img_mat[max(0, y0 - bia):min(n1, y0 + bia + 1),
                  max(0, x0 - bia):min(n1, x0 + bia + 1)]  # 矩阵位置与实际位置的对应(下角标注意)
        real_i_num = 0  # 计数
        i_num = 0
        for j in range(len(small_m)):
            for t in range(len(small_m[0])):
                # 5*5点的总数
                i_num += small_m[j][t]
                # 十字路径的点计数
                if abs(j - 2) == 0 or abs(t - 2) == 0:
                    real_i_num += small_m[j][t]  # or 1
        # min_scale = 0.80   # 最小比例设置！太小的话起不到作用，太大的话会有正确的检测目标也过滤掉
        # 符合点的初步判断条件：最小比例达到min_scale
        if real_i_num > min_num and (real_i_num/i_num) >= min_scale:
            i_x_y_w_h.append(x_y_w_h[i])
            i_flux_list.append(flux_list[i])
    return i_x_y_w_h, i_flux_list

def possible_target_points(kernel_5, img_mat, x_y_w_h, x_y_w_h_f, n1):
    # all_point_x = []  # 用于保存所有检测到的目标点
    # all_point_y = []
    i_x_y_w_h = []  # 保存检测框内该点所在的十字路径上点的x值
    i_x_y_w_h_f = []  # 保存检测框内该点所在的十字路径上点的x值
    # 针对暗源的阈值
    if n1 == 512:
        min_scale = 0.4
        min_corrcoef = 0.12
        photometry_value_t = 14
    else:
        # 针对亮源的阈值——前两个阈值设置小一点，减轻作用！
        min_scale = 0.36
        min_corrcoef = 0.08
        photometry_value_t = 16
    psf_col = kernel_5.reshape(kernel_5.size, order='C')
    for i in range(len(x_y_w_h)):
        x1 = max(int(x_y_w_h[i][0] - x_y_w_h[i][2] / 2), 0)
        y1 = max(int(x_y_w_h[i][1] - x_y_w_h[i][3] / 2), 0)
        x2 = min(int(x_y_w_h[i][0] + x_y_w_h[i][2] / 2), n1 - 1)
        y2 = min(int(x_y_w_h[i][1] + x_y_w_h[i][3] / 2), n1 - 1)
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
            # small_m = resize(img_mat[max(0, int(y1)):min(n1, int(y2)),
            #           max(0, int(x1)):min(n1, int(x2))], (5, 5))          # 将检测框resize为5*5大小
            # small_m = resize(img_mat[y1:y2, x1:x2], (5, 5))          # 将检测框resize为5*5大小
            small_m = resize(img, (5, 5))          # 将检测框resize为5*5大小
            small_m_col = small_m.reshape(small_m.size, order='C')
            corrcoef = abs(np.corrcoef(small_m_col, psf_col)[0, 1])
            # # 验证正确的点的相关数据！
            # small_m = img_mat[max(0, y0 - bia):min(n1, y0 + bia + 1),
            #           max(0, x0 - bia):min(n1, x0 + bia + 1)]  # 矩阵位置与实际位置的对应(下角标注意)
            # 条件3
            for j in range(len(small_m)):
                for t in range(len(small_m[0])):
                    # 5*5点的总数
                    i_num += small_m[j][t]
                    # 十字路径的点计数
                    if abs(j - 2) == 0 or abs(t - 2) == 0:
                        real_i_num += small_m[j][t]  # or 1
        except ValueError:
            print(x_y_w_h[i])
        # min_scale = 0.80   # 最小比例设置！太小的话起不到作用，太大的话会有正确的检测目标也过滤掉
        # 符合点的初步判断条件：最小比例达到min_scale
        if i_num > 0 and corrcoef >= min_corrcoef and (real_i_num/i_num) >= min_scale and photometry_value >= photometry_value_t:
            i_x_y_w_h.append(x_y_w_h[i])
            i_x_y_w_h_f.append(x_y_w_h_f[i])
    return i_x_y_w_h, i_x_y_w_h_f

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("GPU {} will be used\n".format("0"))

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
    load_name = os.path.join(input_dir,
                             'fpn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print(load_name)

    pascal_classes = np.asarray(['__background__', 'EP'])
    # initilize the network here.
    # class-agnostic 方式只回归2类bounding box，即前景和背景
    if args.net == 'res101':
        fpn = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fpn = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fpn = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        # 到了pdb.set_trace()那就会定下来，就可以看到调试的提示符(Pdb)了
        pdb.set_trace()

    fpn.create_architecture()
    fpn.cuda()
    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    fpn.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fpn.cuda()

    # 对dropout和batch normalization的操作在训练和测试的时候是不一样的
    # pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    fpn.eval()

    start = time.time()
    # max_per_image = 2500
    webcam_num = args.webcam_num
    # Set up webcam or get image directories

    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)  # 应该就是判断要不要自己用电脑录视频
        num_images = 0
    else:  # 如果不用电脑录视频，那么就读取image路径下的图片
        # imglist = os.listdir(args.image_dir)
        import glob

        fits_list = os.path.join(args.image_dir, '*_1_*.fits')
        imglist = sorted(glob.glob(fits_list),
                         key=lambda x: int(re.findall("[0-9]+", x.split('_')[-1])[0]))  # 按照数字大小排序！
        num_images = len(imglist)

    print('Loaded Photo: {} images.'.format(num_images))

    flux_interval = np.linspace(0, 10, 41)         #5,6
    all_recall = np.zeros(len(flux_interval))
    all_flux_internal_num = np.zeros(len(flux_interval))
    precision_all = 0
    all_ture_num = 0
    while (num_images >= 0):
        box_list = list()
        all_xy_point_list = []
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1
        psf_matrix = np.load('../psf_matrix_512.npy')
        kernel_5 = resize(psf_matrix, (5, 5))
        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            # ret 为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            im_in = np.array(frame)
        # Load the demo image
        else:
            im_file = os.path.join(args.image_dir, imglist[num_images])
            img_fits = fits.open(im_file, ignore_missing_end=True)[0].data
            # img_fits_cv = ndimage.convolve(img_fits, kernel / np.max(kernel))
            ### use log transpoze
            # img_fits = np.log(1 + np.abs(img_fits))
            max_value = np.max(img_fits)
            min_value = np.min(img_fits)
            mean_value = np.mean(img_fits)
            var_value = np.var(img_fits)
            im_in = (img_fits - mean_value) / (max_value - min_value)
        # im_in = (im - mean_value)/var_value

        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)  # 扩展为三维数据
        # rgb -> bgr
        # line[:-1]其实就是去除了这行文本的最后一个字符（换行符）后剩下的部分。
        # line[::-1]字符串反过来 line = "abcde" line[::-1] 结果为：'edcba'
        im = im_in[:, :, ::-1]

        blobs, im_scales = _get_image_blob(im)  ##图片变换 该文件上面定义的函数，返回处理后的值 和尺度
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
        _, _, _, _, _ = fpn(im_data, im_info, gt_boxes, num_boxes)

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


        for j in range(1, len(pascal_classes)):  # 对每个类别进行遍历画框
            inds = torch.nonzero(scores[:, j] > args.thresh).view(-1)  # 返回不为0的索引
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
                keep = nms(cls_dets, 0.1)    # 0.1
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

        result_save_path = im_file[:-4].rstrip(".") + ".save.txt"
        result_path = im_file[:-4].rstrip(".") + "_det.txt"
        result_path_all = im_file[:-4].rstrip(".") + "_det_all.txt"
        if not box_list:
            np.savetxt(result_path, np.array([0]), fmt="%.8f")  # 若未检测到，则写入-1
        else:
            box_np = np.concatenate(box_list, axis=0)  # 按照列进行拼接！也可以是box_list
            np.savetxt(result_path, box_np, fmt="%.8f")
            # save_mat = np.array(save_mat)  # 按照列进行拼接！也可以是box_list
            point_list_2, box_list_flux_2 = get_xy_flux(box_np, img_fits)
            # 储存可能的目标点的信息-对检测到的结果进行筛选
            # point_list_2, box_list_flux_2 = possible_target_points(kernel_5, img_fits, point_list_2, box_list_flux_2, 256)
            if point_list_2:
                all_xy_point_list = cal_initial_xy(point_list_2, all_xy_point_list, box_list_flux_2, 256)
        if all_xy_point_list:
            all_box_np = np.array(all_xy_point_list)  # 所有最终检测结果保存
            np.savetxt(result_path_all, all_box_np, fmt="%.2f")
        else:
            all_box_np = np.array([0])
            np.savetxt(result_path_all, all_box_np)  # 若未检测到，则写入0

        # 计算某一张图像中检测到的正确目标数量、flux每个区间中正确目标的数量、检测到的数量以及flux每个区间的本该找到的数量
        true_num, recall_internal_num, precision_num, flux_internal_num = cal_indicators_new(result_save_path, all_box_np, flux_interval)
        # true_num, recall_internal_num, precision_num, flux_internal_num = cal_indicators(result_save_path, box_np, img_fits, flux_interval)

        all_ture_num += true_num  # 记录检测目标实际找到的总的正确目标的数量
        all_recall += recall_internal_num  # 记录检测目标在flux每个区间中实际找到的正确目标的数量
        all_flux_internal_num += flux_internal_num  # 记录flux每个区间的本该找到的数量
        precision_all += precision_num  # 记录总的检测到的数量
        if num_images % 50 == 0:
            print(all_ture_num)
            print(all_recall)
            print(all_flux_internal_num)
            print(precision_all)

        if args.vis and webcam_num == -1:
            if all_box_np.any():
            # if box_np.any():
                result_path = imglist[num_images][:-4] + "_det.jpg"
                im2show = (np.log(img_fits + 1))  # .astype(np.uint8)  均匀化
                # scale = 16
                # im2show = vis_detections(im2show, all_box_np, scale)  # 经过NMS之后的进一步置信度阈值设置-显示用！
                im2show = vis_detections(im2show, class_name, box_np, 0.5)
                cv2.imwrite(result_path, im2show/np.max(im2show)*255)  # 写入图片内容到本地
                result_path_plt = im_file[num_images][:-4] + "_detection.jpg"
                plt.figure()
                plt.imshow(np.log(im2show/np.max(im2show)+9)*255)
                plt.savefig(result_path_plt)
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

    # 评价指标：
    try:
        recall = all_recall / all_flux_internal_num  # 两个数组的除法！
    except ZeroDivisionError:
        print("含有分母为0，改用依次遍历区间！")
        recall = np.ones(len(flux_interval))  # 初始化
        for j in range(len(flux_interval)):
            if all_flux_internal_num[j]:
                recall[j] = all_recall[j] / all_flux_internal_num[j]
    precision = all_ture_num / precision_all
    print('session{}_{}_indicators: \n recall = {}\n precision = {:.2f}\n'.format(
        args.checksession, args.thresh, recall, precision,))
    # 绘制曲线
    # plt.figure(figsize=(12, 12))
    # bar_width = 0.5
    bar_width = 1
    plt.figure()
    plt.bar(flux_interval, recall, bar_width, align='edge', color='r')  # align='center '
    plt.xlabel('flux_interval')
    plt.ylabel('recall')
    # plt.plot(flux_interval, recall, color="red", label="recall-img")
    # plt.legend(loc='best')
    plt.title('indicators_img')
    plt.savefig(
        './output/visiual_map_loss/session{}_{}_{}_indicators_img.jpg'
            .format(args.checksession,args.checksession+1, args.thresh))
    plt.close()

    print("completed!!!")