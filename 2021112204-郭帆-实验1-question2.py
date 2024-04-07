import numpy as np

pth = "C:/Users/Dell/Desktop/MLexp/exp1/data/experiment_01_dataset_02.csv" ### 数据集路径
dataset = np.loadtxt( pth, delimiter=',', dtype=float)  ### 使用numpy读入数据集
### 查看数据集
print(dataset.shape)
print(dataset)

def TP (data , col1 , col2) :
    """
    计算真正例数目
    :param data: 数据集
    :param col1: 真实类别所在列
    :param col2: 模型预测类别所在列
    :return: 真正例数目
    """
    sum = 0
    for i in range(data.shape[0]) :
        if data[i][col1] == 1 and data[i][col2] == 1 :
            sum += 1

    return sum

def TN(data, col1, col2):
    """
    计算真反例数目
    :param data: 数据集
    :param col1: 真实类别所在列
    :param col2: 模型预测类别所在列
    :return: 真反例数目
    """
    sum = 0
    for i in range(data.shape[0]) :
        if data[i][col1] == 0 and data[i][col2] == 0 :
            sum += 1

    return sum

def FP(data, col1, col2):
    """
    计算假正例数目
    :param data: 数据集
    :param col1: 真实类别所在列
    :param col2: 模型预测类别所在列
    :return: 假正例数目
    """
    sum = 0
    for i in range(data.shape[0]) :
        if data[i][col1] == 0 and data[i][col2] == 1 :
            sum += 1

    return sum

def FN(data, col1, col2):
    """
    计算假反例数目
    :param data: 数据集
    :param col1: 真实类别所在列
    :param col2: 模型预测类别所在列
    :return: 假反例数目
    """
    sum = 0
    for i in range(data.shape[0]) :
        if data[i][col1] == 1 and data[i][col2] == 0 :
            sum += 1

    return sum

def recall (tp , fn) :
    """
    计算召回率
    :param tp: 模型真正例数量
    :param fn: 模型假反例数量
    :return: 召回率
    """
    return  tp * 100 / (tp + fn)

def precision (tp , fp) :
    """
    计算精确率
    :param tp: 模型真正例数量
    :param fp: 模型假正例数量
    :return: 精确率
    """
    return  tp * 100 / (tp + fp)

def f1 (data , tp , tn) :
    """
    计算F1分数
    :param data: 数据集
    :param tp: 模型真正例数量
    :param tn: 模型真反例数量
    :return: F1分数
    """
    return 2 * tp / (data.shape[0] + tp - tn)

### 对答案进行输出
print('*' * 10)
print("模型一的TP = ",TP(dataset , 1 , 2))
print("模型二的TP = ",TP(dataset , 1 , 3))
print("模型三的TP = ",TP(dataset , 1 , 4))
print('*' * 10)
print("模型一的FP = ",FP(dataset , 1 , 2))
print("模型二的FP = ",FP(dataset , 1 , 3))
print("模型三的FP = ",FP(dataset , 1 , 4))
print('*' * 10)
print("模型一的TN = ",TN(dataset , 1 , 2))
print("模型二的TN = ",TN(dataset , 1 , 3))
print("模型三的TN = ",TN(dataset , 1 , 4))
print('*' * 10)
print("模型一的FN = ",FN(dataset , 1 , 2))
print("模型二的FN = ",FN(dataset , 1 , 3))
print("模型三的FN = ",FN(dataset , 1 , 4))
print('*' * 10)
print("模型一的recall = ",recall(TP(dataset , 1 , 2), FN(dataset , 1 , 2)))
print("模型二的recall = ",recall(TP(dataset , 1 , 3), FN(dataset , 1 , 3)))
print("模型三的recall = ",recall(TP(dataset , 1 , 4), FN(dataset , 1 , 4)))
print('*' * 10)
print("模型一的precision = ",precision(TP(dataset , 1 , 2),FP(dataset , 1 , 2)))
print("模型二的precision = ",precision(TP(dataset , 1 , 3),FP(dataset , 1 , 3)))
print("模型三的precision = ",precision(TP(dataset , 1 , 4),FP(dataset , 1 , 4)))
print('*' * 10)
print("模型一的F1 score = ",f1(dataset,TP(dataset , 1 , 2),TN(dataset , 1 , 2)))
print("模型二的F1 score = ",f1(dataset,TP(dataset , 1 , 3),TN(dataset , 1 , 3)))
print("模型三的F1 score = ",f1(dataset,TP(dataset , 1 , 4),TN(dataset , 1 , 4)))


