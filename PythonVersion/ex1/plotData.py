import numpy as np
import matplotlib.pyplot as plt

from computeCost import *

def plot_regression(theta,x,y):
    """根据参数画出拟合的直线"""
     #画出散点图
    plt.scatter(x[:,1],y,c='red',edgecolors='none',s=10)

    #坐标轴设置
    plt.title('Linear Regression',fontsize=20)
    plt.xlabel('Population of City in 10,000s',fontsize=15)
    plt.ylabel('Profit in $10,000s',fontsize=15)

    #画出拟合直线
    line_x = np.linspace(0,25,100)
    line_x = np.c_[np.ones(100),line_x]
    line_y = np.dot(line_x,theta)
    plt.plot(line_x[:,1],line_y,'b')


def plot_cost_descent(theta_list,x,y):
    n = len(theta_list)

    cost_list = []
    for theta in theta_list:
        cnt_cost = cost(theta,x,y)
        cost_list.append(cnt_cost)
    
    x = list(range(n))
    plt.plot(x,cost_list,'b')
    plt.title('Cost Function',fontsize=20)
    plt.xlabel('Iterations',fontsize=15)
    plt.ylabel('Value of Cost Function',fontsize=15)