"""返回cost和gradient"""

import numpy as np

from sigmoid import *

def cost(theta,X,y):
    """返回cost"""
    #样本数m
    m = y.size

    #当前参数下，对于每个样本的预测结果（是种类1的概率）
    h_theta = sigmoid_matrix(np.dot(theta,X.T))

    total_cost = 0.
    for i in range(m):
        if y[i] == 1:
            total_cost -= np.log(h_theta[i])
        elif y[i] == 0:
            total_cost -= np.log(1-h_theta[i])
    total_cost = total_cost/m

    return total_cost


def gradient(theta,X,y):
    """返回gradient"""
    #样本数m
    m = y.size

    #当前参数下，对于每个样本的预测结果（是种类1的概率）
    h_theta = sigmoid_matrix(np.dot(theta,X.T))

    cnt_gradient = np.zeros(theta.shape)
    for i in range(m):
        cnt_gradient += (h_theta[i]-y[i])*X[i,:]
    cnt_gradient = cnt_gradient/m

    return cnt_gradient