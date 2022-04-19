# -*- coding: utf-8 -*-
# @Time : 2022/4/15 10:54
# @Author : lwb
# @File : deal_init_data.py

import time
from astropy.io import fits
import numpy as np
import os
import pandas as pd
from skimage.transform import rescale, resize
from scipy import ndimage
import cv2
import re
import glob


class generate_multi_size_data(object):
    '''[summary]

    [description]

    Arguments:
        object_size {[type]} -- [description]
        search_size {[type]} -- [description]
    '''

    def __init__(self, n_max_1, n_max_2):
        super(generate_multi_size_data, self).__init__()
        self.n_max_1 = n_max_1  # 大目标小图
        self.n_max_2 = n_max_2  # 小目标大图
        # 加载PSF信息
        self.psf_matrix = np.load('psf_matrix_512.npy')

    # 读取fits图像！
    def read_fits_data(self, path_filename):
        hdu1 = fits.open(path_filename)[1].data
        print("before filter:", len(hdu1))
        xyf = np.c_[hdu1.field('RAWX'), hdu1.field('RAWY'), hdu1.field('PI')]
        filter_xyf = np.array(list(filter(lambda t: 50 <= t[2] <= 400, xyf)))
        n = len(filter_xyf)
        print("after filter:", n)
        return filter_xyf

    # 构造矩阵！
    def creat_matrix_1(self, img_xyf, n1):
        point_matrix = np.zeros((n1, n1))
        x1 = ((img_xyf[:, 0] - 1) // (4096 / n1)).astype(int)  # 向下取整！压缩16倍！math.floor()
        y1 = ((img_xyf[:, 1] - 1) // (4096 / n1)).astype(int)
        for xi, yi in zip(x1, y1):
            point_matrix[yi][xi] += 1
        return point_matrix

    # 构造矩阵！
    def creat_matrix_2(self, img_xyf, n):
        x1 = ((img_xyf[:, 0] - 1) // (4096 / n)).astype(int)  # 向下取整！压缩16倍！math.floor()
        y1 = ((img_xyf[:, 1] - 1) // (4096 / n)).astype(int)
        point_matrix = np.zeros((n, n))
        for xi, yi in zip(x1, y1):
            point_matrix[yi][xi] += 1
        return point_matrix

    def norm_psf(self, img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalization
        return img

    def drop_noise_method(self, img, kernel):
        # 直接将只有一个点的噪点去除
        img = np.where(img > 1, img, 0)  # 满足大于等于1的值保留，不满足的设为0
        # 二维卷积
        den_img = cv2.filter2D(img, -1, kernel / np.max(kernel))  # -1：保证处理之后的图像深度保持一致
        # # 高斯
        # den_img = gaussian_filter(img, sigma=1)
        return den_img

    def get_xy_flux(self, det, img):

        x1 = det[:, 0].astype(np.int32)
        x2 = det[:, 2].astype(np.int32)
        y1 = det[:, 1].astype(np.int32)
        y2 = det[:, 3].astype(np.int32)
        flux_pre = []
        xy_point = []
        for i in range(len(det)):
            flux_pre.append(np.max(img[y1[i]:y2[i], x1[i]:x2[i]]))  # 将原图中框内最大的像素值作为flux值！
            index = np.argmax(img[y1[i]:y2[i], x1[i]:x2[i]])
            w = x2[i] - x1[i]
            h = y2[i] - y1[i]
            x0 = x1[i] + index % w
            y0 = y1[i] + index // w
            xy_point.append((x0, y0, w, h))
        return xy_point, flux_pre

    def cal_initial_xy(self, xy_p, xy_p_new, flux_list, flag):
        '''
        计算原始的中心坐标，同时去除重复检测的目标
        :param xy_p:
        :param xy_p_new:
        :param flux_list:
        :param flag: 尺度大小的标志
        :return: 更新的检测结果
        '''
        xy_p = np.array(xy_p) * (4096 / flag)
        flux_np = np.array(flux_list).reshape(-1, 1)
        xy_f = np.concatenate((xy_p, flux_np), axis=1).tolist()
        if xy_p_new:
            for i in range(len(xy_p_new)):
                xy_f = list(
                    filter(lambda x: abs(x[0] - xy_p_new[i][0]) > 16 * 1 or abs(x[1] - xy_p_new[i][1]) > 16 * 1,
                           xy_f))  # 16可以更改！
            xy_p_new.extend(xy_f)
        else:
            xy_p_new.extend(xy_f)
        return xy_p_new

    # 产生模拟图像！更合理！
    def data_generate(self, all_xy_point_list, s, flag):
        mask = np.zeros((s * 2, s * 2))  # mask作为模板，用以提取目标，做目标分割！
        psf_max = np.max(resize(self.psf_matrix, (s, s), anti_aliasing=True))
        for point_flux in all_xy_point_list:
            point = point_flux[:2]
            flux = point_flux[-1]
            mask_tem = np.zeros((s * 2, s * 2))  # mask作为模板，用以提取目标，做目标分割！
            scale = int(flux / psf_max)  # 计算flux值与PSF的倍数
            psf_matrix_new = self.psf_matrix.copy()
            if scale:
                # 放缩用的是opencv 的resize函数，采用的是线性插值
                psf_matrix_new = cv2.resize(psf_matrix_new, (s * scale, s * scale))
                psf_matrix_new = psf_matrix_new[(len(psf_matrix_new) // 2 - s // 2):(len(psf_matrix_new) // 2 + s // 2),
                                 (len(psf_matrix_new) // 2 - s // 2):(len(psf_matrix_new) // 2 + s // 2)]
            psf_matrix_new = self.norm_psf(cv2.resize(psf_matrix_new, (s, s)))
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

    def cv_light(self, point_matrix, all_xy_point_list):
        '''

        :param point_matrix: 256*256
        :return:
        '''
        # 开始进行预检测
        kernel_21 = resize(self.psf_matrix, (21, 21), anti_aliasing=True)
        kernel = resize(kernel_21, (3, 3))  # 或者为（5，5）

        # 腐蚀
        erosion = cv2.erode(point_matrix, kernel)
        # 卷积：使得图像更具具备psf特性
        # kernel = resize(kernel, (3, 3))
        filter_img = ndimage.convolve(erosion, kernel / np.max(kernel))  # -1：保证处理之后的图像深度保持一致
        _, thresh_img = cv2.threshold(filter_img, 300, 255,
                                      cv2.THRESH_BINARY)  # 图像的二值化  cv.THRESH_BINARY_INV+cv.THRESH_OTSU = 1+8=9
        dealed_img = cv2.dilate(thresh_img, kernel_21)
        contours, hierarchy = cv2.findContours(dealed_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        xy_point = []
        for i, contour in enumerate(contours):
            # 删除边界点
            contour = np.reshape(contour, (-1, 2))
            n = len(contour)
            while n:
                n -= 1
                a = contour[n]
                if (0 in a) or ((self.n_max_1 - 1) in a):
                    contour = np.delete(contour, n, 0)  # axis = 0：arr按行删除
            contour = np.reshape(contour, (-1, 1, 2))
            contours[i] = contour  # 重新赋值！
            x, y, w, h = cv2.boundingRect(contour)  # contour是一个轮廓点集合
            if w != 0 and h != 0:
                if x <= 0 or y <= 0:
                    print(x, y, w, h)
                elif x + w > self.n_max_1 or y + h > self.n_max_1:
                    print(x, y, w, h)
                else:
                    xy_point.append((x, y, x + w, y + h))

        # 若是检测到极亮源，则将其从mask掉
        if xy_point:
            box_np_1 = np.array(xy_point)  # 按照列进行拼接！也可以是box_list
            # 搜寻亮源所在的位置
            xy_point_list, flux_list = self.get_xy_flux(box_np_1, point_matrix)
            all_xy_point_list = self.cal_initial_xy(xy_point_list, all_xy_point_list, flux_list, self.n_max_1)
            flag = 0  # 两次系数乘积
            point_matrix_simulate = self.data_generate(all_xy_point_list, self.n_max_1, flag)
            # 构造掩膜
            # 满足大于1（可以更改）的值保留，不满足的设为0, 模拟的数据（去除的更彻底）
            point_matrix_simulate = np.where(point_matrix_simulate > 1, point_matrix_simulate, 0)
            point_matrix_simulate = np.where(point_matrix_simulate == 0, 1, 0)
            img_1 = np.multiply(point_matrix, point_matrix_simulate)  # 点乘做mask
            # img1 = self.drop_noise_method(img_1, kernel)  # 是否进行去噪处理？
        else:
            # 是否卷积：使得图像更具具备psf特性
            img_1 = point_matrix
            # img1 = self.drop_noise_method(point_matrix, kernel)

        return img_1, all_xy_point_list

    def filter_label_1(self, img_mat, df_label_1, xy_point_list):
        '''
        以图1（256）的图像光子点的数量为准，计算图1和图2的标签
        :param img_mat:
        :param df_label_1:
        :param xy_point_list:
        :return:
        '''
        # 先对一些极亮源的标签进行过滤
        df_label_2_x = (df_label_1['x'] / (4096 / self.n_max_1)).values
        df_label_2_y = (df_label_1['y'] / (4096 / self.n_max_1)).values
        df_label_2_flux = df_label_1['flux/mCrab'].values
        df_label_2 = np.c_[df_label_2_x, df_label_2_y, df_label_2_flux]
        # 依据点的分布来选择想要的暗源目标！
        bia = 2  # 设置计数范围(2*bb)*(2*bb)
        # 删除后的标签坐标
        filter_xyf_1 = []
        filter_xyf_2 = []

        # 过滤极亮源的标签——处于2个像素范围之内
        range_th = 2
        xy_flux = df_label_2.tolist()  # 变成list——可迭代
        if xy_point_list:
            for i in range(len(xy_point_list)):
                xy_flux = list(
                    filter(lambda x: abs(x[0] - xy_point_list[i][0]) > range_th or abs(
                        x[1] - xy_point_list[i][1]) > range_th,
                           xy_flux))
        n_real = len(xy_flux)
        # 遍历标签
        for i in range(n_real):
            y0, x0 = int(xy_flux[i][1]), int(xy_flux[i][0])
            small_m = img_mat[max(0, y0 - bia):min(n1, y0 + bia + 1),
                      max(0, x0 - bia):min(n1, x0 + bia + 1)]  # 矩阵位置与实际位置的对应(下角标注意)
            count_small_m = 0  # 计数
            for j in range(len(small_m)):
                for t in range(len(small_m[0])):
                    # 十字路径的点计数
                    if abs(j - 2) == 0 or abs(t - 2) == 0:
                        count_small_m += small_m[j][t]  # or 1
            if count_small_m >= 24:  # 亮源：20-1200
                filter_xyf_1.append([xy_flux[i][0], xy_flux[i][1], xy_flux[i][2]])
            # elif count_small_m >= 8 and count_small_m < 24:  # 暗源：5-20，对应坐标乘以2
            elif count_small_m >= 8 and count_small_m < 24:  # 暗源：5-20，对应坐标乘以2
                filter_xyf_2.append([xy_flux[i][0] * 2, xy_flux[i][1] * 2, xy_flux[i][2]])
        return filter_xyf_1, filter_xyf_2

    def get_data(self, path_name, save_path):

        # 类别号
        class_num = 100
        all_xy_point_list = []

        # path_name = path + '/eventfile'
        pattern = re.compile(r'\d+')  # 查找数字
        basename_file = path_name.split('/')[-1]
        basename = pattern.findall(basename_file)[0]  # .split('wxt')[0]
        i = pattern.findall(basename_file)[1]

        # 生成二维数据
        imgxyf = self.read_fits_data(path_name)
        point_mat_1 = self.creat_matrix_1(imgxyf, self.n_max_1)  # 已经变为小图处理128*128！
        point_mat_2 = self.creat_matrix_2(imgxyf, self.n_max_2)  # 已经变为大图处理512*512

        # 读取标签数据
        read_path = os.path.join("./obsinfo", 'obsinfo' + basename + '.csv')
        df_label = pd.read_csv(read_path)
        df_label_i = df_label[(df_label['cmosnum'] == int(i))]  # .index.tolist()

        # 保存极亮源中心点
        point_mat_1_new, all_xy_point_list = self.cv_light(point_mat_1, all_xy_point_list)  # 进行极亮源检测并去除极亮源的图像
        point_mat_2_new = point_mat_2  # 点乘做mask# 构造掩膜

        # 生成list标签文件----list文件需要经过处理从才行！
        save_file_list_1 = os.path.join(save_path, basename + '_' + str(1) + '_' + i + '.list')
        save_file_list_2 = os.path.join(save_path + '-2', basename + '_' + str(2) + '_' + i + '.list')

        # 过滤并读取亮源和一般源的标签！——图像取去除之后的图像：更加贴近实际
        filter_xyf_1, filter_xyf_2 = self.filter_label_1(point_mat_1_new, df_label_i, self.n_max_1, all_xy_point_list)
        fp1 = open(save_file_list_1, 'w+')
        if len(filter_xyf_1):
            filter_xyf_1 = np.vstack(filter_xyf_1)
            # 产生亮源的模拟图像
            df_label_1to2_xy = filter_xyf_1[:, :2].tolist()
            flux_list = filter_xyf_1[:, 2].tolist()
            # df_label_1to2_xy = n_max_2/n_max_1*filter_xyf_1[:, :2]   # 按照比例调整！
            flag1 = 1  # 1 or 0
            # 更新检测到的目标位置
            all_xy_point_list = self.cal_initial_xy(df_label_1to2_xy, all_xy_point_list, flux_list,
                                                    self.n_max_1)  # 还是256的尺度
            point_matrix_simulate_2 = self.data_generate(all_xy_point_list, self.n_max_2, flag1)

            # 构造掩膜
            # 满足大于0.2的值保留，不满足的设为0, 模拟的数据（去除的更彻底）
            point_matrix_simulate_2 = np.where(point_matrix_simulate_2 > 0.2, point_matrix_simulate_2, 0)
            point_matrix_simulate_2 = np.where(point_matrix_simulate_2 == 0, 1, 0)
            point_mat_2_new = np.multiply(point_mat_2, point_matrix_simulate_2)  # 点乘做mask# 构造掩膜

            # 写入文件！
            for x_value, y_value, flux in filter_xyf_1.tolist():  # 去重操作set！
                fp1.write("{} {} {} {}\n".format(class_num, x_value, y_value, flux))  # 严谨一点：位置作为标签值，不得马虎
            #  标签去除的更彻底一些！
            fp2 = open(save_file_list_2, 'w+')
            # 判断是否为空
            if len(filter_xyf_2):
                filter_xyf_2 = np.vstack(filter_xyf_2)
                for x_value, y_value, flux in filter_xyf_2.tolist():  # 去重操作set！
                    fp2.write("{} {} {} {}\n".format(class_num, x_value, y_value, flux))  # 严谨一点：位置作为标签值，不得马虎
            else:
                fp2.write("{} {} {} {}\n".format(class_num, -1, -1, -1))
            fp2.close()
        else:
            #  标签去除的更彻底一些！
            # filter_xyf_2 = filter_label_2(point_mat_2, df_label_i, n_max_2)
            fp2 = open(save_file_list_2, 'w+')
            if len(filter_xyf_2):
                filter_xyf_2 = np.vstack(filter_xyf_2)
                for x_value, y_value, flux in filter_xyf_2.tolist():  # 去重操作set！
                    fp2.write("{} {} {} {}\n".format(class_num, x_value, y_value, flux))  # 严谨一点：位置作为标签值，不得马虎
            else:
                fp2.write("{} {} {} {}\n".format(class_num, -1, -1, -1))
            fp2.close()
            fp1.write("{} {} {} {}\n".format(class_num, -1, -1, -1))
        fp1.close()

        # 归一化处理！
        # point_mat_norm_1 = norm(point_mat_1)
        # point_mat_norm_2 = norm(point_mat_2)

        # save fits file
        fitsfilename_1 = os.path.join(save_path, basename + '_' + str(1) + '_' + i + '.fits')
        fitsfilename_2 = os.path.join(save_path + '-2', basename + '_' + str(2) + '_' + i + '.fits')

        if os.path.exists(fitsfilename_1):
            os.remove(fitsfilename_1)
        grey = fits.PrimaryHDU(point_mat_1_new)
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(fitsfilename_1)
        if os.path.exists(fitsfilename_2):
            os.remove(fitsfilename_2)
        grey2 = fits.PrimaryHDU(point_mat_2_new)  # 保存Mask亮源之后的图像
        greyHDU = fits.HDUList([grey2])
        greyHDU.writeto(fitsfilename_2)


    def get_data_test(self, path_name, save_path):

        # path_name = path + '/eventfile'
        pattern = re.compile(r'\d+')  # 查找数字
        basename_file = path_name.split('/')[-1]
        basename = pattern.findall(basename_file)[0]  # .split('wxt')[0]
        i = pattern.findall(basename_file)[1]
        imgxy = self.read_fits_data(path_name)
        point_mat_1 = self.creat_matrix_1(imgxy, self.n_max_1)  # 已经变为小图处理256*256！
        point_mat_2 = self.creat_matrix_2(imgxy, self.n_max_2)  # 已经变为大图处理512*512！
        # 归一化处理！
        # point_mat_norm_1 = norm(point_mat_1)
        # point_mat_norm_2 = norm(point_mat_2)
        # save fits file
        fitsfilename_1 = os.path.join(save_path, basename + '_' + str(1) + '_' + i + '.fits')
        fitsfilename_2 = os.path.join(save_path, basename + '_' + str(2) + '_' + i + '.fits')
        if os.path.exists(fitsfilename_1):
            os.remove(fitsfilename_1)
        grey = fits.PrimaryHDU(point_mat_1)
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(fitsfilename_1)
        if os.path.exists(fitsfilename_2):
            os.remove(fitsfilename_2)
        grey = fits.PrimaryHDU(point_mat_2)  # 保存Mask亮源之后的图像
        greyHDU = fits.HDUList([grey])
        greyHDU.writeto(fitsfilename_2)

    def generate_two_stage_data(self, train, path):
        # train控制是否进行训练: 0 or 1
        # 针对新数据：没有区别！
        evtlist = glob.glob(os.path.join(path, 'ep119*wxt*po_cl.evt'))  # 加载第一种类型的图像
        train_save_path = './traindata300-300'
        test_save_path = './testdata300-300'
        # val_save_path1 = './detect_1/validation'
        # val_save_path2 = './detect_2/validation'
        all_val_save_path = './detect_1and2/validation'  # 只适合于联合检测！
        num = len(evtlist)
        train_evtlist = evtlist[: int(0.8 * num)]
        test_evtlist = evtlist[int(0.8 * num + 1):]
        # imglist = sorted(evtlist,
        #                  key=lambda x: int(
        #                      re.findall("[0-9]+", x.split('wxt')[-1].split('po_cl.evt')[0])[0]))  # 按照数字大小排序！  [0]:取列表的第一个值，int:转为整数类型

        train_num = len(train_evtlist)
        test_num = len(test_evtlist)
        if train:
            while train_num:
                train_num -= 1
                train_evt_file = train_evtlist[train_num]
                self.get_data(train_evt_file, train_save_path)

            while test_num:
                test_num -= 1
                test_evt_file = test_evtlist[test_num]
                self.get_data(test_evt_file, test_save_path)

        else:
            while num:
                num -= 1
                evt_file = evtlist[num]
                self.get_data_test(evt_file, all_val_save_path)


if __name__ == '__main__':
    start = time.time()
    # 原始数据
    path = "./init_data"
    size1 = 256
    size2 = 512
    train = 0   # or 1
    datas = generate_multi_size_data(size1, size2)
    datas.generate_two_stage_data(train, path)
    end = time.time()
    print('times:', end - start)
