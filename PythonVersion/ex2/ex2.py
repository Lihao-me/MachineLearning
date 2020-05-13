import numpy as np
import scipy.optimize as opt

from plotData import *
from costFunction import *
from plot_decision_boundary import *

# ================第一部分：载入数据和画图===============
data = np.loadtxt('ex2data1.txt',delimiter=',',usecols=(0,1,2))
x = data[:,0:2]
y = data[:,2]
plot_dataset1(x,y)
plt.show()

# ================第二部分：优化=========================
#数据数量m
m = y.size

#加上一列1到x
X = np.c_[np.ones(m),x]

#初始参数
n = X.shape[1]
ini_theta = np.zeros(n)
print("ini_cost:   ",cost(ini_theta,X,y))
print("ini_grad:   ",gradient(ini_theta,X,y),"\n\n")

#使用scipy中的bfgs优化方法
opt_theta,opt_cost,*unused = opt.fmin_bfgs(f=cost,x0=ini_theta,fprime=gradient,args=(X,y),full_output=True)

# ================第三部分：画出decision boundry=============
plot_dataset1(x,y)
plot_decision_boundary(opt_theta)
plt.show()