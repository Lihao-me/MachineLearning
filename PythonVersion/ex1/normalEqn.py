import numpy as np

def normal_eqn(x,y):
    #用normal equation求解theta
    theta = np.linalg.inv(np.dot(x.T,x)).dot(x.T).dot(y)
    
    return theta
