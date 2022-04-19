from DataPreprocess import Data_preprocess, MyDataset
# from torch.utils.data import Dataset, DataLoader
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# mechine
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
# confusion matrix plotting
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix


def main_machine(save_name):
    object_size = 13
    search_size = 5
    star_path = './ep_data/ep_real_data'
    noise_path = './ep_data/ep_noise_data'
    star_csv_path = './ep_data/star_small_threshold/'
    noise_csv_path = './ep_data/target/'
    star_num = None
    noise_num = None
    train_size = 0.8
    test_size = 0.2
    random_state = 123

    # 数据预处理
    test = Data_preprocess(object_size, search_size,
                           star_path, noise_path, star_csv_path, noise_csv_path,
                           star_num, noise_num,
                           train_size, test_size, random_state,
                           'train')
    # None)

    # 属性再划分
    test.enhance_switch = False
    test.mutil_process_switch = False
    test.rotangstep = 1
    test.angle = 0
    test.noise_step = 1

    # 数据集划分——乱序划分
    (Xtrain, Ytrain), (Xtest, Ytest) = test.mutil_2tensor()

    # 使用随机森林算法进行训练并生成预处理结果
    # save_name = save_name.replace('DecisionTree', 'RandomForest')
    train_start = time.time()
    reg = RandomForestClassifier(n_estimators=10).fit(Xtrain, Ytrain)
    predicted_reg = reg.predict(Xtest)
    cm_reg = confusion_matrix(predicted_reg, Ytest)
    accuracy_RF = accuracy_score(predicted_reg, Ytest)
    print("Accuracy_RF = ", accuracy_RF)
    fig = plt.figure()
    _ = plot_confusion_matrix(cm_reg, colorbar=True)  # class_names=class_names
    plt.title("Confusion_matrix of RandomForestClassifier")
    plt.savefig(os.path.join("./result/machine_RF", "confusion_matrix_RF.jpg"))
    # plt.show()
    joblib.dump(value=reg, filename=save_name)

    return train_start

if __name__ == '__main__':
    # main()
    save_name = "./result/machine_RF/RandomForest_for_ep_11.model"
    start = time.time()
    train_start = main_machine(save_name)
    end = time.time()
    print("训练时间：", end-train_start)
    print("总时间：", end-start)
