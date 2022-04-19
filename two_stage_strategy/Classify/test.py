# from PIL import Image, ImageDraw
import numpy as np
import os
import glob
from skimage.transform import resize
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

def use_classify(path, Model_PATH):
	npylist = glob.glob(os.path.join(path, '*.npy'))  # 加载第一种类型的图像
	n = len(npylist)
	print("筛选前：", n)
	object_size = 13
	target_result = []
	false_result = []

	classifier_model = joblib.load(Model_PATH)
	test_data = np.zeros((n, object_size * object_size))
	# Ytest = np.zeros(n)
	Ytest = np.ones(n)

	while n:
		n -= 1
		# 准备数据——4维
		data = resize(np.load(npylist[n]), (object_size, object_size))
		test_data[n, :] = data.reshape(data.size, order='C')

	# 模型测试
	predicted = classifier_model.predict(test_data)
	accuracy_RF = accuracy_score(predicted, Ytest)
	print("Accuracy_RF = ", accuracy_RF)

	for i in range(len(predicted)):
		if predicted[i] == 0:
			target_result.append(npylist[i])
		else:
			false_result.append(npylist[i])

	return target_result, false_result



if __name__ == '__main__':
	path = "./ep_data/ep_real_data"
	# path = "./ep_data/ep_noise_data"
	Model_PATH = os.path.join("./result/machine_RF", "RandomForest_for_ep.model")
	# t = use_classify_deep(path, Model_PATH)
	targ , false_t = use_classify(path, Model_PATH)
	print("筛选后：", len(targ))
	for i in range(9):
		sub_data = resize(np.load(targ[i]), (13, 13))
		print(sub_data)
	for i in range(9):
		sub_data = resize(np.load(false_t[i]), (13, 13))
		print(sub_data)

