import numpy as np
import matplotlib.pyplot as plt

from plotData import *
from mapFeature import *

def plot_decision_boundary(theta):
    """为 ex2.py 画出 decision boundary"""
    boud_x = np.array([10,120])
    boud_y = (-theta[1]*boud_x-theta[0])/theta[2]
    plt.plot(boud_x,boud_y,'b-')


def plot_decision_boundary_reg(theta):
    """画出非线性的decision boundary"""
    #划分网格
    grid_x = np.linspace(-1,1.2,100)
    grid_y = np.linspace(-1,1.2,100)
    xx,yy = np.meshgrid(grid_x,grid_y)

    #计算z：各个节点处，z>=0 代表判断为类型1
    z = np.dot(map_feature(xx.ravel(),yy.ravel()),theta)
    z = z.reshape(xx.shape)

    plt.contour(xx,yy,z,0)


#====================================================以下部分为淘汰的代码==========================

def which_class(theta,x1,x2):
    """基于模型得到的参数，判断一个点(x1,x2)是哪一类的"""
    x1 = np.array([x1])
    x2 = np.array([x2])
    z = np.dot(theta,map_feature(x1,x2).ravel())
    if z < 0:
        return 0
    return 1


def is_on_decision_boundary(theta,x1,x2,delta):
    """判断一个点(x1,x2)是否在非线性的decision boundary上"""
    #如果一个点，它左边delta/2和右边delta/2那两个点不是同一类的，认为这个点在decision boundary上
    #同理，如果上面delta/2和下面delta/2那两个点不是同一类的，也认为它在之上
    if which_class(theta,x1-delta/2,x2) != which_class(theta,x1+delta/2,x2):
        return 1
    elif which_class(theta,x1,x2-delta/2) != which_class(theta,x1,x2+delta/2):
        return 1
    return 0

def plot_decision_boundary_reg_unused(theta):
    """为 ex2_reg.py 画出 decision boundary"""
    #设定一个范围，在这个矩形范围里画图
    min_x = -1.0
    max_x = 1.2
    min_y = -1.0
    max_y = 1.2

    #划分网格，然后对网格里每一个节点进行判断它是否在decision boundary上
    delta = 0.05 #一个网格的长和宽
    num_x = int((max_x-min_x)/delta+1)
    num_y = int((max_y-min_y)/delta+1)

    grid_x = np.linspace(min_x,max_x,num=num_x)
    grid_y = np.linspace(min_y,max_y,num=num_y)

    boundary_points_x = []
    boundary_points_y = []
    for i in range(num_x):
        for j in range(num_y):
            if is_on_decision_boundary(theta,grid_x[i],grid_y[j],delta):
                boundary_points_x.append(grid_x[i])
                boundary_points_y.append(grid_y[j])

    #画图
    plt.scatter(boundary_points_x,boundary_points_y,c='blue',marker='o',s=5)