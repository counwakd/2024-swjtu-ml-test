import numpy as np
from matplotlib import pyplot as plt

# 读取所用到的数据集
train_data = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp3/data/experiment_03_training_set.csv', delimiter=',')
test_data = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp3/data/experiment_03_testing_set.csv',delimiter=',')

def TP (y_true , pre) :
    """
    计算TP值
    :param y_true: 真实的标签值
    :param pre:   预测的标签值
    :return:  TP值
    """
    num = 0
    for i in range(len(y_true)) :
        if y_true[i] == 1 and pre[i] == 1 : num = num + 1

    return num


def FP(y_true, pre):
    """
    计算FP值
    :param y_true: 真实的标签值
    :param pre:   预测的标签值
    :return:  FP值
    """
    num = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and pre[i] == 1: num = num + 1

    return num


def TN(y_true, pre):
    """
    计算TN值
    :param y_true: 真实的标签值
    :param pre:   预测的标签值
    :return:  TN值
    """
    num = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and pre[i] == 0: num = num + 1

    return num


def FN(y_true, pre):
    """
    计算FN值
    :param y_true: 真实的标签值
    :param pre:   预测的标签值
    :return:  FN值
    """
    num = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and pre[i] == 0: num = num + 1

    return num

class Logistic:
    """
    逻辑回归模型
    """
    def __init__(self,dimension,lr=0.1, lambda_val=0.01, iter=100):
        """
        对模型的基本参数进行设置
        :param dimension: 特征维度
        :param lr: 学习率
        :param lambda_val: 正则化系数
        :param iter: 迭代次数
        """
        self.lr = lr
        self.lambda_val = lambda_val
        self.iter = iter
        self.weights = np.zeros((dimension,1))   ### 将weight矩阵初始化为全0
        self.bias = 0       ### 将bias初始化为0

    def sigmoid(self, x):
        ### 实现sigmoid函数
        return 1 / (1 + np.exp(-x))

    def loss_fn(self, input, target):
        ### 使用交叉熵作为损失函数并添加正则化防止过拟合
        pre = self.sigmoid(np.dot(input, self.weights) + self.bias)
        regularization_term = (self.lambda_val / (2 * input.shape[0])) * np.sum(np.square(self.weights))  ### 正则化项计算
        loss = (-1 / input.shape[0]) * np.sum(target * np.log(pre) + (1 - target) * np.log(1 - pre)) + regularization_term  ### 二元交叉熵 + 正则化

        return loss

    def backward(self, input, y):
        ### 使用梯度下降对参数进行优化求解
        m, n = input.shape
        loss = 0

        loss = self.loss_fn(input,y)
        pre = self.sigmoid(np.dot(input, self.weights) + self.bias)
        dw = (1 / m) * np.dot(input.T, (pre - y)) + (self.lambda_val / m) * self.weights   ### 对weigh矩阵进行求导
        db = (1 / m) * np.sum(pre - y)             ### 对bias进行求导
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

        return loss

    def forward (self, input , threshold):
        """
        计算预测标签
        :param input: 输入的特征矩阵
        :param threshold: 分类阈值
        :return: 预测标签
        """
        output = self.sigmoid(np.dot(input, self.weights) + self.bias)
        output = (output > threshold).astype(int)

        return output

x = train_data[:, : train_data.shape[1] - 1]   ### 特征矩阵
label = train_data[:,train_data.shape[1] - 1].reshape(-1,1)   ### 正式标签

model = Logistic(x.shape[1])   ### 创建逻辑回归模型
predict = model.forward(x , 0.5)   ### 计算预测值

train_loss = []   ### 存储每一轮的损失函数值

for it in range (100) :    ### 进行100轮迭代
    loss = 0
    train_loss.append(model.backward(x , label))

### 使用测试集进行模型评估
x = test_data[:, : train_data.shape[1] - 1]
label = test_data[:,train_data.shape[1] - 1].reshape(-1,1)
predict = model.forward(x,0.5)
print(f"TP : {TP(label , predict)}  ,  FN : {FN(label , predict)}")   ### 计算TP ， FN
print(f"FP : {FP(label , predict)}  ,  TN : {TN(label , predict)}")   ### 计算TN ， FP
print('错误率 :  ' , 1 - (TP(label , predict) + TN(label , predict)) / x.shape[0])   ### 计算错误率
print('精度 :  ' , (TP(label , predict) + TN(label , predict)) / x.shape[0])          ### 计算精度
print('查准度 :  ' , TP(label , predict) / (TP(label , predict) + FP(label , predict)))     ### 计算查准率
print('查全率 :    ' , TP(label , predict) / (TP(label , predict) + FN(label , predict)))   ### 计算recall
print('F1分数 :    ' , 2 * TP(label ,predict) / (x.shape[0] + TP(label , predict) - TN(label , predict)))   ### 计算F1_score

### 对训练过程的损失函数进行绘图展示
x = range(100)
plt.plot(x , train_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Traing Loss')

plt.show()
