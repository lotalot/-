import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets
from torchvision import transforms
from sklearn import svm
from sklearn import preprocessing
from sklearnex import patch_sklearn
import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import learning_curve

'''
    @brief  加载MNIST数据集并转换格式成二值图
    @param train: 是否为训练集
    @param data_enhance: 是否进行数据增强
    @return 二值图集和标签集
'''


def LoadMnistDataset(train=True, data_enhance=False):
    mnist_set = datasets.MNIST(root="./MNIST", train=train, download=True)
    x_, y_ = list(zip(*([(np.array(img), target) for img, target in mnist_set])))
    sets_raw = []
    sets_r20 = []
    sets_invr20 = []
    y = []
    y_r20 = []
    y_invr20 = []
    sets = []
    matrix_r20 = cv2.getRotationMatrix2D((14, 14), 25, 1.0)
    matrix_invr20 = cv2.getRotationMatrix2D((14, 14), -25, 1.0)
    select = 0
    for idx in range(len(x_)):
        # 对图像进行二值化以及数据增强
        _, img = cv2.threshold(x_[idx], 255, 255, cv2.THRESH_OTSU)
        sets_raw.append(np.array(img.data).reshape(784))
        y.append(y_[idx])
        if data_enhance:
            if select % 2 == 0:
                img_r20 = ~cv2.warpAffine(~img, matrix_r20, (28, 28), borderValue=(255, 255, 255))
                sets_r20.append(np.array(img_r20.data).reshape(784))
                y_r20.append(y_[idx])
            else:
                img_invr20 = ~cv2.warpAffine(~img, matrix_invr20, (28, 28), borderValue=(255, 255, 255))
                sets_invr20.append(np.array(img_invr20.data).reshape(784))
                y_invr20.append(y_[idx])
            select += 1

    # 数据增强
    sets = sets_raw + sets_r20 + sets_invr20
    sets = np.array(sets)
    print(sets.shape)
    if data_enhance:
        y = y + y_r20 + y_invr20
    return sets, y

'''
    @brief  保存SVM模型

    @param svc_model: SVM模型 
    @param file_path: 模型保存路径，默认为./SVC

    @return None
'''

def SaveSvcModel(svc_model, file_path="./SVC"):
    with open(file_path, 'wb') as fs:
        pickle.dump(svc_model, fs)

'''
     @brief  加载SVM模型

     @param file_path: 模型保存路径，默认为./SVC

     @return SVM模型
'''

def LoadSvcModel(file_path="./SVC"):
    if not os.path.exists(file_path):
        assert "Model Do Not Exist"
    with open(file_path, 'rb') as fs:
        svc_model = pickle.load(fs)
    return svc_model


'''
     @brief  训练SVM模型

     @param c: SVM参数C
     @param enhance: 是否进行数据增强

     @return acc: 在测试集上的准确率
             svc_model: SVM模型
'''


def TrainSvc(c, enhance):
    # 读取数据集，训练集及测试集
    images_train, targets_train = LoadMnistDataset(train=True, data_enhance=enhance)
    images_test, targets_test = LoadMnistDataset(train=False, data_enhance=enhance)

    # 训练
    svc_model = svm.SVC(C=c, kernel='rbf', decision_function_shape='ovr')
    svc_model.fit(images_train, targets_train)

    # 在测试集上测试准确度

    res = svc_model.predict(images_test)
    correct = (res == targets_test).sum()
    accuracy = correct / len(images_test)
    print(f"测试集上的准确率为{accuracy * 100}%")
    return svc_model


'''
     @brief  预处理比较粗的字体

     @param image: 输入图像
     @:param show: 是否显示预处理后的图像
     @:param thresh: 二值化阈值

     @return 预处理后的图像数据
'''


def PreProcessFatFont(image, show=False):
    # 白底黑字转黑底白字
    pre_ = ~image

    # 转单通道灰度
    pre_ = cv2.cvtColor(pre_, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # resize后添加黑色边框，亲测可提高识别率
    pre_ = cv2.resize(pre_, (112, 112))
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    back = np.zeros((300, 300), np.uint8)
    back[29:141, 29:141] = pre_
    pre_ = back

    if show:
        cv2.imshow("show", pre_)
        cv2.waitKey(0)

    # 做一次开运算(腐蚀 + 膨胀)
    kernel = np.ones((2, 2), np.uint8)
    pre_ = cv2.erode(pre_, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    pre_ = cv2.dilate(pre_, kernel, iterations=1)

    # 第二次resize
    pre_ = cv2.resize(pre_, (56, 56))
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # 做一次开运算(腐蚀 + 膨胀)
    kernel = np.ones((2, 2), np.uint8)
    pre_ = cv2.erode(pre_, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    pre_ = cv2.dilate(pre_, kernel, iterations=1)

    # resize成输入规格
    pre_ = cv2.resize(pre_, (28, 28))
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # 转换为SVM的输入格式
    pre_ = np.array(pre_).flatten().reshape(1, -1)
    return pre_


'''
     @brief  预处理细的字体

     @param image: 输入图像
     @param show: 是否显示预处理后的图像
     @param thresh: 二值化阈值


     @return 预处理后的图像数据
'''


def PreProcessThinFont(image, show=False):
    # 白底黑字转黑底白字
    pre_ = ~image

    # 转灰度图
    pre_ = cv2.cvtColor(pre_, cv2.COLOR_BGR2GRAY)

    # 增加黑色边框
    pre_ = cv2.resize(pre_, (112, 112))
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    back = np.zeros((170, 170), dtype=np.uint8)  # 这里不指明类型会导致后续矩阵强转为float64，无法使用大津法阈值
    back[29:141, 29:141] = pre_
    pre_ = back

    if show:
        cv2.imshow("show", pre_)
        cv2.waitKey(0)

    # 对细字体先膨胀一下
    kernel = np.ones((3, 3), np.uint8)
    pre_ = cv2.dilate(pre_, kernel, iterations=2)

    # 第二次resize
    pre_ = cv2.resize(pre_, (56, 56))

    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # 做一次开运算(腐蚀 + 膨胀)
    kernel = np.ones((2, 2), np.uint8)
    pre_ = cv2.erode(pre_, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    pre_ = cv2.dilate(pre_, kernel, iterations=1)

    # resize成输入规格
    pre_ = cv2.resize(pre_, (28, 28))
    _, pre_ = cv2.threshold(pre_, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # 转换为SVM输入格式
    pre_ = np.array(pre_).flatten().reshape(1, -1)

    return pre_


'''
     @brief  在空白背景上显示提取出的roi

     @param res_list: roi列表

     @return None
'''


def ShowRoi(res_list):
    back = 255 * np.ones((1000, 1500, 3), dtype=np.uint8)
    # 图片x轴偏移量
    tlx = 0

    for roi in res_list:
        if tlx + roi.shape[1] > back.shape[1]:
            break
        # 每次在原图上加上一个roi
        back[0:roi.shape[0], tlx:tlx + roi.shape[1], :] = roi
        tlx += roi.shape[1]

    cv2.imshow("show", back)
    cv2.waitKey(0)


'''
     @brief  寻找数字轮廓并提取roi

     @param src: 输入图像
     @param thin: 是否为细字体
     @param thresh: 二值化阈值

     @return roi列表
'''


def FindNumbers(src, thin=True):
    # 拷贝
    dst = src.copy()
    paint = src.copy()
    roi = src.copy()
    dst = ~dst

    # 预处理
    paint = cv2.resize(paint, (448, 448))
    dst = cv2.resize(dst, (448, 448))

    # 记录缩放比例,后来看这一步好像没啥意义
    fx = src.shape[1] / 448
    fy = src.shape[0] / 448

    # 转单通道
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # 边缘检测后二值化，直接二值化的话由于采光不同的原因灰度直方图峰与峰之间可能会差距过大，导致二值图的分割不准确
    # 而边缘检测对像素突变更加敏感，因此采用Canny边缘检测后二值化
    cv2.Canny(dst, 200, 200, dst)

    # 对于平常笔写的字太细，膨胀一下
    if thin:
        kernel = np.ones((5, 5), np.uint8)
        dst = cv2.dilate(dst, kernel, iterations=1)

    # 寻找外围轮廓
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 提取roi
    roi_list = []
    rect_list = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        if not ((rect[2] * rect[3] < 400 or rect[2] * rect[3] > 448 * 448 / 2.5) or (rect[3] < rect[2])):
            cv2.rectangle(paint, rect, (255, 0, 0), 1)
            x_min = rect[0] * fx
            x_max = (rect[0] + rect[2]) * fx
            y_min = rect[1] * fy
            y_max = (rect[1] + rect[3]) * fy
            roi_list.append(roi[int(y_min):int(y_max), int(x_min):int(x_max)].copy())
            rect_list.append(rect)
    return paint, roi_list, rect_list


'''
     @brief  以txt形式显示数据

     @param data: 数据集

     @return None   
'''


def ShowDataTxt(data):
    print("----------------------------------------------------------")
    for i in range(28):
        for j in range(28):
            print(0 if data[0][i * 28 + j] == 255 else 1, end='')
        print('\n')
    print("----------------------------------------------------------")


if __name__ == "__main__":
    # 加载
    patch_sklearn()
    model_path = "./SVC_C1_enhance.pkl"

    if os.path.exists(model_path):
        print("Model Exist, Load Form Serialization")
        model = LoadSvcModel(model_path)
    else:
        print("Model Do Not Exist, Train")

        # 训练
        model = TrainSvc(1, False)

        # 保存
        SaveSvcModel(model, model_path)

    # 测试
    paint, nums, rects = FindNumbers(cv2.imread("E:\\c.jpg"))
    predict_nums = []
    for img in nums:
        data = PreProcessThinFont(img, show=False)
        # ShowDataTxt(data)
        predict_nums.append(model.predict(data)[0])
    for i in range(len(predict_nums)):
        cv2.putText(paint, str(predict_nums[i]), (rects[i][0], rects[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1)
    recognized_student_id = ''.join(map(str, predict_nums))
    print(recognized_student_id, 'recognized student ID')
    cv2.imshow("show", paint)
    cv2.waitKey(0)