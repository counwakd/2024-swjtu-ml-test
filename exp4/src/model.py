import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as ply
import matplotlib
from sklearn.tree import DecisionTreeClassifier , plot_tree

train_data = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp4/data/experiment_04_training_set.csv' , delimiter= ',')  ### 读取训练集
test_data = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp4/data/experiment_04_testing_set.csv' ,delimiter= ',')   ### 读取测试集

### 调用DecisionTreeClassifier接口创建决策树模型
model = DecisionTreeClassifier(random_state=1 , criterion= 'entropy' , max_depth= 3)

### 提取训练集中的特征和标签
train_x = train_data[: ,0 : -1].reshape(train_data.shape[0] , -1)
train_y = train_data[: , -1].reshape (train_data.shape[0] , -1)

### 提取测试集中的特征和标签
test_x = test_data[: , 0:-1].reshape(test_data.shape[0] , -1)
test_y = test_data[: , -1].reshape (test_data.shape[0] , 1)

### 对模型进行训练
model.fit(train_x , train_y)

### 获得模型得分
score = model.score(test_x , test_y)
print(score)

### 对模型进行测试
predict = model.predict(test_x)  ### 得到模型预测值

### 计算模型在测试集上的准确率
acc = 0
for i in range(test_y.shape[0]) :
    if predict[i] == test_y[i,0] : acc += 1   ### 计算预测正确的数量

acc = acc / test_data.shape[0]          ### 计算最终的准确率
print(acc)

### 可视化产生的决策树模型
rose = [] ### 给每个特征字段赋值
for i in range(train_data.shape[1] - 1) :
    rose.append(f'feature_{i + 1}')

plot_tree(model , feature_names= rose , class_names= ['0','1','2'], filled= True)
plt.title('criterion = entropy && max_depth = 1') ### 可替换max_depth

ply.show()
