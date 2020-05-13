import numpy as np

def feature_normalize(x):
    """对数据集做 normalization : avg返回各个feature的均值，std返回各个feature的标准差...
    ...normalized_x 返回normalize后的x"""

    #获取featrue个数 n
    n = x.shape[1]

    #计算 avg，std
    avg = np.mean(x,axis=0)
    std = np.std(x,axis=0)

    #计算 noramlized_x
    normalized_x = x.copy()
    for i in range(n):
        #对第 i 个feature进行处理
        normalized_x[:,i] = (x[:,i]-avg[i])/std[i]

    return avg,std,normalized_x