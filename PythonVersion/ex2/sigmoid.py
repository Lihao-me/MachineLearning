import numpy as np

def sigmoid_single_value(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_matrix(x):
    """x的类型是np.array"""
    size_tuple = x.shape
    res = np.asarray(x,dtype=float)


    if len(size_tuple) == 1:
        # x 是 vector 时：
        for i in range(size_tuple[0]):
            res[i] = sigmoid_single_value(x[i])

    elif len(size_tuple) == 2:
        # x 是 matrix 时：
        for i in range(size_tuple[0]):
            for j in range(size_tuple[1]):
                res[i][j] = sigmoid_single_value(x[i][j])

    return res

