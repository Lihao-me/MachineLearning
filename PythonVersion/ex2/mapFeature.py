"""接受x，返回新的feature（排成一个矩阵X，每一行对应一个样本）"""

import numpy as np

def map_feature(x1,x2):
    #样本数m
    m = x1.shape[0]

    result = np.ones(m)
    for degree in range(6):
        degree += 1
        for i in range(degree+1):
            result = np.c_[result,(x1**(degree-i))*(x2**i)]

    return result