import PsfExtract_modify
from torch.utils.data import Dataset
from multiprocessing import Pool
import numpy as np
import random
import cv2
import math
import sys
import os
import glob
from skimage.transform import resize
from sklearn.model_selection import train_test_split


class Data_read(object):
    '''[summary]

    [description]

    Arguments:
        object_size {[type]} -- [description]
        search_size {[type]} -- [description]
    '''

    def __init__(self, object_size, search_size):
        super(Data_read, self).__init__()
        self.object_size = object_size
        self.search_size = search_size

    def data_load(self, array_path, csv_path, flag):
        '''[summary]

        [description] 数据加载核心！

        Arguments:
            image_path {[type]} -- [description]
            csv_path {[type]} -- [description]

        Returns:
            [type] -- [description]
        '''
        # 读取数据
        npy_list = os.path.join(array_path, '*.npy')  # 加载第一种类型的图像
        # fits_list2 = os.path.join(args.image_dir, '*_2_*.fits')  # 加载第二种类型的图像，也可以直接将1改为2
        imglist = glob.glob(npy_list)
        n = len(imglist)
        objects = np.zeros((n, self.object_size, self.object_size))
        for i in range(n):
            img = np.load(imglist[i])
            objects[i, :, :] = resize(img, (self.object_size, self.object_size), preserve_range=True)
        if flag == 0:
            # objects_label = np.zeros([np.shape(objects)[0]], dtype=np.int32)
            objects_label = np.zeros((np.shape(objects)[0], 1), dtype=np.int32)
            return objects, objects_label

        elif flag == 1:
            # objects_label = np.ones([np.shape(objects)[0]], dtype=np.int32)
            objects_label = np.ones((np.shape(objects)[0], 1), dtype=np.int32)
            return objects, objects_label

        else:
            print('No Label !!!')
            return objects

    # return objects, objects_label

    # return objects, objects_label

    def data_load_debug_mode(self, image_path, csv_path, flag):
        '''[summary]

        [description]

        Arguments:
            image_path {[type]} -- [description]
            csv_path {[type]} -- [description]

        Returns:
            [type] -- [description]
        '''
        objects = PsfExtract_modify.Psfobtain_auto_center(image_path, csv_path, self.object_size, self.search_size)[0]
        if flag == 0:
            objects_label = np.zeros([np.shape(objects)[0]], dtype=np.int32)
        # objects_label = np.zeros((np.shape(objects)[0], 1), dtype=np.int32)
        elif flag == 1:
            objects_label = np.ones([np.shape(objects)[0]], dtype=np.int32)
        # objects_label = np.ones((np.shape(objects)[0], 1), dtype=np.int32)
        return objects, objects_label

    def rotate(self, src, angle, scale=1):
        '''[summary]

        [description]
        '''
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # rot_mat = cv2.getRotationMatrix2D((nw, nh), angle, scale)
        # rot_move = np.dot(rot_mat, np.array([(nw-w), (nh-h), 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(src, rot_mat, (int(math.ceil(w)), int(math.ceil(h))))

    def data_enhanse(self, image_path, csv_path, rotangstep, angle, flag):
        '''[summary]

        [description]

        Returns:
            [type] -- [description]
        '''
        objects = np.zeros([1, self.object_size, self.object_size])
        psfs = self.data_load(image_path, csv_path, 2)
        orgnum = np.shape(psfs)[0]
        print('增强前shape', psfs.shape)

        # rotangstep = 4
        for i in range(orgnum):
            # temstar = psfs[:, :, i]
            temstar = psfs[i, :, :]
            for j in range(rotangstep):
                # middle = rotation1.rotation_objects(temstar, 90 * (j), outsize=starsize)
                middle = self.rotate(temstar, angle * (j), scale=1)
                # print('middle:', middle.shape)
                # Remove the images with bad rotation
                objects = np.concatenate([objects, middle[np.newaxis, :, :]], axis=0)
        objects = objects[1:, :, :]
        print('增强后shape', objects.shape)

        if flag == 0:
            # objects_label = np.zeros([np.shape(objects)[0]], dtype=np.int32)
            objects_label = np.zeros((np.shape(objects)[0], 1), dtype=np.int32)
            return objects, objects_label
        elif flag == 1:
            # objects_label = np.ones([np.shape(objects)[0]], dtype=np.int32)
            objects_label = np.ones((np.shape(objects)[0], 1), dtype=np.int32)
            return objects, objects_label
        else:
            print('请输入正确的标签值！')
            sys.exit()
    # return objects, objects_label


class Data_preprocess(Data_read):
    """docstring for Data_preprocess"""

    def __init__(self, object_size, search_size, star_path, noise_path, star_csv_path, noise_csv_path, star_num,
                 noise_num, train_size, test_size, random_state, mode):
        '''
        Arguments:
            object_size {[type]} -- [description]
            search_size {[type]} -- [description]
            star_path {[type]} -- [description]
            noise_path {[type]} -- [description]
            star_csv_path {[type]} -- [description]
            noise_csv_path {[type]} -- [description]
        '''
        super(Data_preprocess, self).__init__(object_size, search_size)
        self.star_path = star_path
        self.noise_path = noise_path
        self.star_csv_path = star_csv_path
        self.noise_csv_path = noise_csv_path

        self.star_num = star_num
        self.noise_num = noise_num

        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.mode = mode

        self.enhance_switch = False
        self.mutil_process_switch = True

        self.star_step = 1
        self.noise_step = 1

        self.rotangstep = 4
        self.angle = 90

    def mutil_process_load(self):
        '''[summary]

        [description]

        Returns:
            [type] -- [description]
        '''
        p = Pool(2)
        if self.enhance_switch is False:
            file = [[self.star_path, self.star_csv_path, 0],
                    [self.noise_path, self.noise_csv_path, 1]]

            if self.mode == 'train':
                result = p.starmap(self.data_load, file)
            else:
                result = p.starmap(self.data_load_debug_mode, file)

        else:
            file = [[self.star_path, self.star_csv_path, self.rotangstep, self.angle, 0],
                    [self.noise_path, self.noise_csv_path, self.rotangstep, self.angle, 1]]

            if self.mode == 'train':
                result = p.starmap(self.data_enhanse, file)
            else:
                result = p.starmap(self.data_load_debug_mode, file)

        p.close()
        p.join()

        return result

    def mutil_load(self):
        '''[summary]

        [description]

        Returns:
            [type] -- [description]
        '''
        if self.enhance_switch is False:
            stars = self.data_load(self.star_path, self.star_csv_path, 0)
            noises = self.data_load(self.noise_path, self.noise_csv_path, 1)
        # noises = self.noise_load(self.noise_path, self.noise_csv_path, 1)
        else:
            stars = self.data_enhanse(self.star_path, self.star_csv_path, self.rotangstep, self.angle, 0)
            noises = self.data_load(self.noise_path, self.noise_csv_path, 1)
        # noises = self.noise_load(self.noise_path, self.noise_csv_path, 1)
        # stars = self.data_load(self.star_path, self.star_csv_path, 0)
        # noises = self.data_enhanse(self.noise_path, self.noise_csv_path, self.rotangstep, self.angle, 1)

        return stars, noises

    def define_split(self, objects, objects_label, num, step):
        '''[summary]

        [description]

        Arguments:
            objects {[type]} -- [description]
            objects_label {[type]} -- [description]
            num {[type]} -- [description]

        Returns:
            [type] -- [description]
        '''
        # 更新实际的数量
        if num is None:
            num = np.shape(objects)[0]

        return train_test_split(objects[0:num:step, :, :], objects_label[0:num:step], train_size=self.train_size,
                                test_size=self.test_size, random_state=self.random_state)

    def data_split(self):
        '''[summary]

        [description]

        Returns:
            [type] -- [description]
        '''
        if self.mutil_process_switch is True:
            (Star, Star_label), (Noise, Noise_label) = self.mutil_process_load()
        elif self.mutil_process_switch is False and self.enhance_switch is True:
            (Star, Star_label), (Noise, Noise_label) = self.mutil_load()
        else:
            (Star, Star_label), (Noise, Noise_label) = self.mutil_load()

        print('星数量%d 噪声数量%d' % (Star.shape[0], int(Noise.shape[0] / self.noise_step)))

        Strain, Stest, StrainL, StestL = self.define_split(Star, Star_label, self.star_num, self.star_step)
        Dtrain, Dtest, DtrainL, DtestL = self.define_split(Noise, Noise_label, self.noise_num, self.noise_step)

        AllTraindata = np.concatenate([Strain, Dtrain], axis=0)
        AllTestdata = np.concatenate([Stest, Dtest], axis=0)
        train_label = np.concatenate([StrainL, DtrainL], axis=0)
        test_label = np.concatenate([StestL, DtestL], axis=0)

        return AllTraindata, train_label, AllTestdata, test_label

    def data_random(self, all_data, all_label, flag):
        '''[summary]

        [description]：打乱顺序

        Arguments:
            all_data {[type]} -- [description]
            all_label {[type]} -- [description]
        '''
        total_num = np.shape(all_data)[0]
        # #######radom ditrubution of data
        random1 = random.sample(range(0, total_num), total_num)
        # change psf distribution
        # psf = np.zeros((total_num, self.object_size, self.object_size), dtype=np.float64)
        # psf_label = np.zeros((total_num), dtype=np.int32)
        psf = np.zeros((total_num, self.object_size * self.object_size), dtype=np.float32)
        psf_label = np.zeros((total_num), dtype=np.float32)
        # psf_label = np.zeros((total_num, 1), dtype=np.float32)
        for r1 in range(total_num):
            c1 = random1[r1]
            data_col = all_data[r1, :, :]
            psf[c1, :] = data_col.reshape(data_col.size, order='C')
            psf_label[c1] = all_label[r1]

        return psf, psf_label

    def mutil_2tensor(self):
        '''[summary]

        [description]

        Returns:
            [type] -- [description]
        '''
        AllTraindata, train_label, AllTestdata, test_label = self.data_split()
        train_result = self.data_random(AllTraindata, train_label, 'train')
        test_result = self.data_random(AllTestdata, test_label, 'test')

        return train_result, test_result


class MyDataset(Dataset):
    '''[summary]

    [description]

    Extends:
        Dataset
    '''

    def __init__(self, datas, labels):
        super(MyDataset, self).__init__()
        self.datas = datas
        self.labels = labels

    def __getitem__(self, item):
        data = self.datas[item, :, :, :]
        label = self.labels[item]
        return (data, label)

    def __len__(self):
        # print(self.datas.size(0))
        # print(self.labels.size(0))
        # return len(self.labels.size(0))
        return self.labels.size(0)
