"""用于数据可视化"""
import numpy as np
import matplotlib.pyplot as plt

def plot_data_points(x,y):
    """画出数据点分布，不同类型用不同颜色"""
    #得到数据的组数
    m = y.size

    type1_x=[]
    type1_y=[]
    type2_x=[]
    type2_y=[]
    for i in range(m):
        if y[i] == 0:
            type1_x.append(x[i,0])
            type1_y.append(x[i,1])
        else:
            type2_x.append(x[i,0])
            type2_y.append(x[i,1])
   
    #画图
    plt.scatter(type1_x,type1_y,c='yellow',marker='o',edgecolors='black',s=15)
    plt.scatter(type2_x,type2_y,c='black',marker='+',edgecolors='black',s=15)

def plot_dataset1(x,y):
    plot_data_points(x,y)
    #坐标轴范围和legend
    plt.axis([10,120,10,120])
    plt.legend(['Not Admitted','Adimitted'],loc=1)

def plot_dataset2(x,y):
    plot_data_points(x,y)
    #坐标轴范围和legend
    plt.axis([-1,1.2,-1,1.2])
    plt.legend(['y=0','y=1'],loc=1)

