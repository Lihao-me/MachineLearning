import numpy as np
import scipy.optimize as opt

from mapFeature import *
from plotData import *
from costFunctionReg import *
from plotDecisionBoundary import *

# ============第一部分：加载数据并画图 =======================
data = np.loadtxt('ex2data2.txt',delimiter=',')
x = data[:,0:2]
y = data[:,2]

plot_dataset2(x,y)
plt.show()
# ============第二部分：优化 ===================
#产生新的feature
X = map_feature(x[:,0],x[:,1])

#初始参数和lambda的值
ini_theta = np.zeros(28)
lambda_val = 1

opt_theta,opt_cost,*unused = opt.fmin_bfgs(f=cost_reg,x0=ini_theta,fprime=gradient_reg,args=(X,y,lambda_val),full_output=1)
# ============第三部分：画出decision boundary==========
plot_dataset2(x,y)
plot_decision_boundary_reg(opt_theta)
plt.show()

plot_dataset2(x,y)
plot_decision_boundary_reg_unused(opt_theta)
plt.show()