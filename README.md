##   1. Introduction

In this project, we use the Faster-RCNN as basic build blocks of source detection algorithm. Meanwhile, since targets with different signal to noise ratios would extend to different scales, we propose a two–step detection strategy. Original images will be zoomed to two images of smaller size: 256×256 pixels (Level-1 image) and 512×512 pixels (level-2 image). Images of 256×256 pixels would be used to detect ultra-bright and bright targets and images of 512 × 512 pixels would be used to detect ordinary targets. In addition, we apply weak restriction conditions to prediction results of each model. Finally, to integrate all detection results, we introduce a random forest based classifier to further screen out false alarms. 

## 2. Installation

The code is in Ubuntu 18 04、pytorch1. 7.1 +cu110 was tested and the following Python dependencies were installed (using`pip install`）：

```
astropy
skimage
scipy
opencv-python
matplotlib
photutils
sklearn
——再加入一些！
```

## 3. Data preprocessing

You can download sample point clouds [HERE](https://drive.google.com/file/d/1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l/view?usp=sharing). Unzip the file under the project root path (`./project/two_stage_strategy/init_data`) and then run (The parameter train is set to 0 by default) :

```
cd two_stage_strategy
python3 deal_init_data.py
```

## 4. Two step target detection algorithm

You can download pre-trained models [HERE](https://drive.google.com/file/d/1oem0w5y5pjo2whBhAqTtuaYuyBu1OG8l/view?usp=sharing). Unzip the file under the project root path (`./project/two_stage_strategy/detect_1and2/model_save/es50/EP_detect`、`./project/two_stage_strategy/Classify/result/machine_RF`)  and then run:

```
cd detect_1and2
python3 demo_fits_soft_nms.py
```

Detection results will be dumped to `detect_1and2/validation`.

Evaluation results will be dumped in the `detect_1and2/output/visiual_map_loss` folder (or any other folder you specify). In default we evaluate with both precision  and recall.

## 5. Train on new data

### 5.1 Data preprocessing 

Set the parameter train in the `deal_init_data.py` to 1 and then  run  `deal_init_data.py `.

###  5.2 Train the bright source detection model and the ordinary source detection model respectively 

Parameter configuration is in the config file`(./detect_1/lib/model/utils/config.py)` and function: `parse_ args()`.

Train bright source detection model separately

```
cd detect_1
python3 train_test_soft_nms.py
```

Train ordinary source detection model separately

```
cd detect_2
python3 train_test_soft_nms.py
```

### 5.3  Test the bright source detection model and the general source detection model respectively 

Prepare the test data and put it into the corresponding directory named`validation` , then load the trained model and run:

Test bright source detection model separately

```
cd detect_1
python3 demo_fits_soft_nms.py
```

Test ordinary source detection model separately

```
cd detect_2
python3 demo_fits_soft_nms.py
```

### 5.4 Train the random forest model

Before training the random forest model,  You need  to set the parameter RF_train in the `detect_1and2/demo_fits_soft_nms.py` to `True` and then run  two-step target detection algorithm to generate training data, as follows:

```
cd detect_1and2
python3 demo_fits_soft_nms.py --RF_train True
cd ..
cd Classify
python3 train.py
```

### 5.5 Test random forest model 

Load the trained random forest model and then run:

```
python3 test.py
```

