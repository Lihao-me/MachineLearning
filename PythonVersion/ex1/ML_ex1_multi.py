import numpy as np

from featureNormalize import *
from gradientDescent import *
from plotData import *
from normalEqn import *

#读入数据
data = np.loadtxt('ex1data2.txt',delimiter=',')

x = data[:,0:2]
y = data[:,2]

m = y.size   #获取样本数
n = x.shape[1]  #feature个数

# Normalization
avg,std,x = feature_normalize(x)

x = np.c_[np.ones(m),x] #加上一列1到x上

#初始参数,学习率,可接受误差，最大迭代次数
ini_theta = np.zeros(n+1)
alpha = 0.003
maxIteration = 100000
acceptable_err = 1e-3

#梯度下降
theta_list = gradient_descent(alpha,ini_theta,maxIteration,acceptable_err,x,y)

##比较梯度下降和normal equation的解
print('梯度下降最终结果：    theta = ',theta_list[-1])
print('Normal Equation结果： theta = ',normal_eqn(x,y))

#可视化
plot_cost_descent(theta_list,x,y)
plt.show()