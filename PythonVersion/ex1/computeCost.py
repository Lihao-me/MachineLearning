import numpy as np

def  cost(theta,x,y):
    """计算对应参数的Cost"""
    #数据集样本数
    m = y.size

    #计算cost
    err_vec = np.dot(x,theta)-y
    cost = np.dot(err_vec,err_vec)/(2*m)

    return cost
