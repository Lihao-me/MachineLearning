import numpy as np
import matplotlib.pyplot as plt

from plotData import *
from gradientDescent import *
from normalEqn import *

#加载数据
data = np.loadtxt('ex1data1.txt',delimiter=',',usecols=(0,1))
x = data[:,0]
y = data[:,1]

m = y.size

x = np.c_[np.ones(m),x] #加上一列1到x

#初始参数,学习率,可接受误差，最大迭代次数
ini_theta = np.array([1.0,1.0])
alpha = 0.01
acceptable_err = 1e-6
maxIteration = 10000

#梯度下降
theta_list = gradient_descent(alpha,ini_theta,maxIteration,acceptable_err,x,y)

#比较梯度下降和normal equation的解
print('梯度下降最终结果：    theta = ',theta_list[-1])
print('Normal Equation结果： theta = ',normal_eqn(x,y))

# 可视化
plot_regression(theta_list[-1],x,y)
plt.show()
plot_cost_descent(theta_list,x,y)
plt.show()

