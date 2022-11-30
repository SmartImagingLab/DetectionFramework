## 1. Introduction

In this project, we integrate several machine learning algorithms to build a target detection framework for data obtained by lobster eye telescopes. Our framework would firstly generate two 2D images with different pixel scales according to positions of photons on the detector. Then an algorithm based on morphological operations and two neural networks would be used to detect candidates of celestial objects with different flux from these 2D images. At last, a random forest algorithm will be used to pick up final detection results from candidates obtained by previous steps. The framework proposed in this paper could be used as references for data processing methods developed for other lobster eye X-ray telescopes.

Next we explain the test use of the code.

## 2.Installation

The code was tested in Ubuntu 18.04, python3.8(cuda:11.0 or 11.1) and pytorch1. 7.1 and the following Python packages were installed (using`pip install`）or  `pip install -r requirements.txt`：

```
torch
astropy
opencv-python  # （cv2）
scikit-image   # （skimage）
matplotlib
photutils
scikit-learn   #（sklearn）
joblib
glob
pandas
easydict

# yaml:can remove from code or install
# pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ pyyaml
# pytorch install
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
```

Please unzip framework.rar and run:

```
cd framework
pip install framework-1.0.tar.gz
```

## 3. Data preprocessing

**If there's data in detect_1and2/validation , please ignore this step.**

Please unzip project.rar. You can  run  deal_init_data.py  to generate new data (The parameter train is set to 0 by default. If you want to retrain the model, set the parameter train  to 1 in deal_init_data.py to generate the corresponding training data) :

```
cd project
python3 deal_init_data.py
```

## 4. The target detection framework

Preparatory work that may be required:

First, go to the  ./lib/model/AlignPool folder to install an environment dependency.

```python
cd detect_1and2
cd ./lib/model/AlignPool
python setup.py install
#The version of torch required is less than 1.11.
```

then you can start the detection program. :

```
cd detect_1and2
python3 demo_fits.py
```

Detection results will be dumped to `detect_1and2/validation`.

Evaluation results will be dumped in the `detect_1and2/output/visiual_map_loss` folder (or any other folder you specify). In default we evaluate with both precision  and recall.

If there is an error in the previous step, go to this step:

```python
cd ./lib/model/soft_nms
python setup_linux.py build_ext --inplace
```

and then:

1. Install gcc:

   ```
   sudo apt-get install gcc
   ```

2. Install g++:

   ```
   sudo apt-get install g++
   ```

3. If it is not successful:

   ```
   sudo apt-get update
   ```

## 5. Train new model

If you want to retrain the new target detection model, you need to train the bright source detection model and the ordinary source detection model respectively.

### 5.1 Data preprocessing

Set the parameter train in the `deal_init_data.py` to 1 and then  run  `deal_init_data.py `.

```
cd ..
python3 deal_init_data.py --train 1
```

### 5.2 Train the bright source detection model and the ordinary source detection model respectively

Parameter configuration is in the config file`(./detect_1/lib/model/utils/config.py)` and function: `parse_ args()`.

Train bright source detection model separately

```
cd detect_1
python3 train_test.py
```

Train ordinary source detection model separately

```
cd detect_2
python3 train_test.py
```

### 5.3  Test the bright source detection model and the ordinary source detection model respectively

Prepare the test data and put it into the corresponding directory named`validation` , then load the trained model and run:

Test bright source detection model separately

```
cd detect_1
python3 demo_fits.py
```

Test ordinary source detection model separately

```
cd detect_2
python3 demo_fits.py
```

### 5.4 Train a random forest model

Before training the random forest model,  You need  to set the parameter RF_train in the `detect_1and2/demo_fits.py` to `True` and then run  two-step target detection algorithm to generate training data, as follows:

```
cd detect_1and2
python3 demo_fits.py --RF_train True
cd ..
cd Classify
python3 train.py
```

### 5.5 Test the RF model

Load the random forest model trained in the previous step into the test.py and then run:

```
python3 test.py
```

## 5.6 Test new target detection framework

Load the bright source detection model, the ordinary source detection model and the random forest model trained in the previous step into the demo_fits.py and then run:

```
cd detect_1and2
python3 demo_fits.py --RF_train False
```

