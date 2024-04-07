import numpy as np

pth = "C:/Users/Dell/Desktop/MLexp/exp1/data/experiment_01_dataset_01.csv"  ### 数据集路径
dataset = np.loadtxt( pth, delimiter=',', dtype=float)      ### 使用numpy读入数据集
print(dataset.shape)      ### 查看数据集的形状
print(dataset)    ### 查看数据集

def MSE (data , col1 , col2) :
    """
    此函数用于计算均方误差
    :param data: 数据集
    :param col1: 真实值所在列
    :param col2: 模型预测值所在列
    :return: 计算出的均方误差
    """
    sum = 0.0
    for i in range(data.shape[0]) :
        sum += (data[i][col1] - data[i][col2]) * (data[i][col1] - data[i][col2])

    return sum / data.shape[0]

def MAE (data , col1 , col2) :
    """
    此函数用于计算平均绝对误差
    :param data: 数据集
    :param col1: 真实值所在列
    :param col2: 模型预测值所在列
    :return: 计算出的平均绝对误差
    """
    sum = 0.0
    for i in range(data.shape[0]):
        sum += np.abs(data[i][col1] - data[i][col2])

    return sum / data.shape[0]

def RMSE (data , col1 ,col2) :
    """
    此函数用于计算均方根误差
    :param data: 数据集
    :param col1: 真实值所在列
    :param col2: 模型预测值所在列
    :return: 计算出的均方根误差
    """
    sum = 0.0
    for i in range(data.shape[0]):
        sum += (data[i][col1] - data[i][col2]) * (data[i][col1] - data[i][col2])

    return np.sqrt(sum / data.shape[0])

print("-" * 10)
### 计算MAE
print("模型一的MAE =  " , MAE(dataset,1,2))
print("模型二的MAE =  " ,MAE(dataset,1,3))
print("模型三的MAE =  " ,MAE(dataset,1,4))
print("-" * 10)
### 计算MSE
print("模型一的MSE =  " ,MSE(dataset,1,2))
print("模型二的MSE =  " ,MSE(dataset,1,3))
print("模型三的MSE =  " ,MSE(dataset,1,4))
print("-" * 10)
### 计算RMSE
print("模型一的RMSE =  " ,RMSE(dataset,1,2))
print("模型二的RMSE =  " ,RMSE(dataset,1,3))
print("模型三的RMSE =  " ,RMSE(dataset,1,4))
