import matplotlib.pyplot as plt
import numpy as np

test_pth = "C:/Users/Dell/Desktop/MLexp/exp2/data/experiment_02_testing_set.csv"  ### 测试集路径
train_pth = "C:/Users/Dell/Desktop/MLexp/exp2/data/experiment_02_training_set.csv"  ### 训练集路径

train_set = np.loadtxt(train_pth, delimiter=',', dtype=float)  ### 读入训练集
test_set = np.loadtxt(test_pth, delimiter=',', dtype=float)  ### 读入测试集


class Linear:
    """
    定义线性回归类形式为 y = wx + b
    """
    def __init__(self, data):
        """
        初始化线性回归模型的权重参数 w，从随机生成的数据中获取维度信息。

        :param data: 训练集数据，numpy数组
        """
        self.weight = np.mat(np.random.random((data.shape[1], 1)))

    def forward(self, input):
        """
        前向传播计算模型的预测值

        :param input: 输入特征数据
        :return: 模型的预测值
        """
        data = np.insert(input, input.shape[1], np.ones(self.weight.shape[0] - input.shape[1]), axis=0)
        output = np.matmul(self.weight.T, data)
        return output

    def backward(self, input, y):
        """
        反向传播更新模型的权重参数

        :param input: 输入特征数据
        :param y: 真实标签数据
        """
        input = np.insert(input, input.shape[1], np.ones(self.weight.shape[0] - input.shape[1]), axis=1)
        self.weight = np.matmul(input.T, input)
        self.weight = np.linalg.inv(self.weight)
        self.weight = np.matmul(np.matmul(self.weight, input.T), y)


# 创建线性回归模型对象
model = Linear(train_set)

# 从训练集中提取特征 x 和标签 y
x = train_set[:, 0].reshape(-1, 1)
y = train_set[:, 1].reshape(-1, 1)

# 使用反向传播算法更新模型参数
model.backward(x, y)

train_loss = []
pre_y = []
x_label = []
# 遍历训练集数据
for i, (x, y_true) in enumerate(train_set):
    x_label.append(x)
    x = x.reshape(-1, 1)
    # 使用训练好的模型进行预测
    y_pre = model.forward(x)
    pre_y.append(y_pre[0][0])
    # 计算训练集上的损失
    train_loss.append((y_true - y_pre[0][0]) * (y_true - y_pre[0][0]))

print("train loss: ", sum(train_loss) / len(train_set))

test_loss = []
# 遍历测试集数据
for i, (x, y_true) in enumerate(test_set):
    x_label.append(x)
    x = x.reshape(-1, 1)
    # 使用训练好的模型进行预测
    y_pre = model.forward(x)
    pre_y.append(y_pre[0][0])
    # 计算测试集上的损失
    test_loss.append((y_true - y_pre[0][0]) * (y_true - y_pre[0][0]))

print("test loss: ", sum(test_loss) / len(test_set))

# 绘制训练集和测试集的散点图，并画出线性回归拟合线
fig = plt.Figure(figsize=(20, 8), dpi=80)
x = train_set[:, 0]
y = train_set[:, 1]
plt.scatter(x, y, label='train_set')   ### 训练集可视化

x = test_set[:, 0]
y = test_set[:, 1]
plt.scatter(x, y, label='test_set')   ### 测试集可视化

plt.plot(x_label, pre_y, label='linear_regression', c='r')  ### 模型可视化

plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression: y = wx + b")

plt.legend()
plt.show()

print("模型的参数为 : ------------\n",model.weight)
