import numpy as np
from matplotlib import pyplot as plt

test_pth = "C:/Users/Dell/Desktop/MLexp/exp2/data/experiment_02_testing_set.csv"  ### 测试数据路径
train_pth = "C:/Users/Dell/Desktop/MLexp/exp2/data/experiment_02_training_set.csv"  ### 训练数据路径

train_set = np.loadtxt(train_pth,delimiter=',', dtype=float)   ### 读入测试集
test_set = np.loadtxt(test_pth,delimiter=',', dtype=float)     ### 读入训练集


class Linear:
    """
    使用闭式解的二次线性回归模型。
    """
    def __init__(self):
        """
        初始化 Linear 类，设定随机权重。
        """
        self.weight = np.mat(np.random.random((3, 1)))

    def backward(self, input, y_train):
        """
        使用闭式解计算权重。

        :param input: 输入数据。
        :param y_train: 目标值。
        """
        X_train_augmented = np.c_[input ** 2, input, np.ones_like(input)]  # 构建增广矩阵，包括 x^2 和 x
        self.weight = np.linalg.inv(X_train_augmented.T.dot(X_train_augmented)).dot(X_train_augmented.T).dot(y_train) ### 计算最终的参数


model = Linear()

### 构建特征和目标值
X_train = train_set[:, 0].reshape(-1,1)
X_test = test_set[:, 0].reshape(-1,1)
y_train = train_set[:, 1].reshape(-1,1)
y_test = test_set[:, 1].reshape(-1,1)

model.backward(X_train , y_train)   ### 构造闭式解

# 构建增广矩阵，包括 x^2 和 x
X_train_augmented = np.c_[X_train**2, X_train, np.ones_like(X_train)]
X_test_augmented = np.c_[X_test**2, X_test, np.ones_like(X_test)]

y_pred_train = X_train_augmented.dot(model.weight)   ### 计算训练集合上的预测值
train_loss = np.mean((y_train - y_pred_train)**2)    ### 计算训练集上的损失函数
print("train loss : " , train_loss)

y_pred_test = X_test_augmented.dot(model.weight)   ### 计算训练集合上的预测值
test_loss = np.mean((y_test - y_pred_test)**2)     ### 计算训练集上的损失函数
print("test loss : " , test_loss)

### 对结果和数据集进行可视化
fig = plt.Figure(figsize=(20,8) , dpi = 80)
x = train_set[:,0]
y = train_set[:,1]
plt.scatter(x,y, label = 'train_set')   ### 训练集可视化

x = test_set[:,0]
y = test_set[:,1]
plt.scatter(x,y, label = 'test_set')   ### 测试集可视化

X = np.insert(X_train , X_train.shape[0] , X_test , axis=0).reshape(-1,)  ### 将训练集和测试集的特征进行拼接
sorted_indices = np.argsort(X)
X = X[sorted_indices].reshape(-1,1)   ### 对不同样本的特征进行排序
x_label = np.c_[X**2, X, np.ones_like(X)]  # 构建增广矩阵，包括 x^2 和 x
pre_y = x_label.dot(model.weight)
x_label = np.insert(X_train , X_train.shape[0] , X_test , axis=0).reshape(-1,)
sorted_indices = np.argsort(x_label)
x_label = x_label[sorted_indices].reshape(-1,1)
plt.plot(x_label , pre_y , label = 'Fitted Model' , c = 'g' , linewidth=3)   ### 画出最终拟合的模型
### 对图像进行标注
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression : y = wx^2 + wx + b")

plt.legend()

plt.show()

print("模型的权重为 : ----------\n" ,model.weight)




