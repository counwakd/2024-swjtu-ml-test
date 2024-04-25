import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as ply
import pandas as pd
import seaborn as sns

train_set = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp4/data/experiment_04_training_set.csv' , delimiter=',')
test_set = np.loadtxt('C:/Users/Dell/Desktop/MLexp/exp4/data/experiment_04_testing_set.csv' , delimiter=',')

train_set = pd.DataFrame(train_set)
test_set = pd.DataFrame(test_set)


fig,axes=ply.subplots(nrows=1,ncols=2,figsize=(9,4))
bplot1=axes[0].boxplot(train_set,
                       vert=True,
                       patch_artist=True,
                       )
bplot2=axes[1].boxplot(test_set,
                       vert=True,
                       patch_artist=True)


colors = ['pink', 'lightblue', 'lightgreen' , 'maroon' , 'y' , 'c']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

for ax in axes:
    ax.set_xlabel('feature') #设置x轴名称
    ax.set_ylabel('label') #设置y轴名称

axes[0].set_title('Train_set Data_distribution')
axes[1].set_title('Test_set Data_distribution')

ply.show()


rose = []
for i in range(train_set.shape[1] - 1) :
    rose.append(f'feature{i + 1}')
rose.append('label')

csy = train_set.set_axis(rose, axis='columns')
cc = test_set.set_axis(rose, axis='columns')

#
# for i in rose :
#     sns.kdeplot(data=csy[i], label="train_data", fill=True)
#     sns.kdeplot(data=cc[i], label="test_data", fill=True)
#     ply.title(f'Data distribution of {i}')
#     ply.legend()
#     ply.show()

num = [0 , 0 , 0]
print(csy.info())
for i in range (train_set.shape[0]) :
    num[int(csy.iloc[i , -1])] += 1

print(num)

x = range(3)
x_1 = [0 , 1 , 2]
# ply.bar(x, num ,  tick_label = x_1)
# plt.xlabel('label')
# plt.ylabel('count')
#
# for x,y in enumerate(num):
#     plt.text(x,y,"%s"%y,ha='center')  #round(y,1)是将y值四舍五入到一个小数位
#
# ply.show()

res = csy.nunique()
res = res[:res.shape[0] - 1]
x = range(res.shape[0])
rose = rose[:len(rose) - 1]

ply.barh(x, res ,  tick_label = rose , color=['r','g','b','r','g','b','r','g','b','r','g','b' , 'r'])
plt.xlabel('count')
plt.ylabel('count')
ply.title('Number_of_values_taken')

ply.show()
