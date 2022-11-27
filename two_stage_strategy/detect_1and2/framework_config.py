# -*- coding: utf-8 -*-
# @Time : 2022/10/8 16:45
# @Author : lwb
# @File : FrameworkConfig.py

import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--s', dest='level_1_size',
                        help='The size of level-1 image',
                        default=256, type=int)  # can alter
    parser.add_argument('--s1', dest='level_2_size',
                        help='The size of level-2 image',
                        default=512, type=int)
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
                        action='store_true')  # 用CUDA
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
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=46, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=90, type=int)  # 可改 fpn_161_72_2395  1-46-90
    parser.add_argument('--checksession_2', dest='checksession_2',
                        help='checksession to load model',
                        default=121, type=int)
    parser.add_argument('--checkepoch_2', dest='checkepoch_2',
                        help='checkepoch to load network',
                        default=13, type=int)
    parser.add_argument('--checkpoint_2', dest='checkpoint_2',
                        help='checkpoint to load network',
                        default=229, type=int)  # 可改 162_26_10548  or 121-13-229
    parser.add_argument('--Model_PATH', dest='Model_PATH',
                        help='checkpoint to load network',
                        default="../Classify/result/machine_RF/RandomForest_for_ep_1.model")  # 可改 RandomForest_for_ep.model
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