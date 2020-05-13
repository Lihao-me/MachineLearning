import numpy as np

def gradient_theta(theta,x,y):
    """返回参数的梯度（其反方向即为下一步迭代的搜索方向）"""
    m=y.size

    gra_theta = np.dot(np.dot(theta.T,x.T),x) - np.dot(y.T,x)
    gra_theta = gra_theta/m

    return gra_theta

def next_parameter(alpha,theta,x,y):
    """返回下一步的参数theta"""
    #样本数
    m = y.size

    #参数迭代
    new_theta=theta-alpha*gradient_theta(theta,x,y)
    
    return new_theta

def gradient_descent(alpha,ini_theta,maxIteration,acceptable_err,x,y):
    """从初始参数开始迭代，返回一个列表，记录迭代过程中每一步的参数theta"""
    theta_list = []
    theta_list.append(ini_theta)

    #梯度下降
    while True:
        cnt_theta = theta_list[-1]
        next_theta = next_parameter(alpha,cnt_theta,x,y)
        theta_list.append(next_theta)

        #结束条件：超过最大迭代次数 or 误差可接受
        if np.linalg.norm(gradient_theta(next_theta,x,y))<acceptable_err:
            print("迭代次数：",len(theta_list))
            break
        if len(theta_list) > maxIteration:
            print("迭代次数过多！")
            break

    return theta_list


