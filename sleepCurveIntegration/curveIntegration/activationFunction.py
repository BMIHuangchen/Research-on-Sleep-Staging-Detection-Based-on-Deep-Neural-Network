from matplotlib import pyplot as plt
import numpy as np
from pylab import *

def relu(x):
    """relu函数"""
    return np.where(x<0,0,x)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def plot_softmax():
    x = np.arange(-10, 10, 0.1)
    fx = softmax(x)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('softmax ')
    plt.plot(x, fx)
    plt.show()

def plot_sigmoid():
    x = np.arange(-10, 10, 0.01)
    fx = sigmoid(x)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('sigmoid ')
    plt.plot(x, fx)
    plt.show()

def plot_relu():
    x = np.arange(-10, 10, 0.01)
    fx = relu(x)
    # ---------------------------------------------
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('relu ')
    plt.plot(x, fx)
    plt.show()

def plot_tanh():
    x = np.arange(-10, 10, 0.01)
    fx = tanh(x)
    ax = plt.gca()  # 得到图像的Axes对象
    ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
    ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
    ax.xaxis.set_ticks_position('bottom')  # 将x轴刻度设在下面的坐标轴上
    ax.yaxis.set_ticks_position('left')  # 将y轴刻度设在左边的坐标轴上
    ax.spines['bottom'].set_position(('data', 0))  # 将两个坐标轴的位置设在数据点原点
    ax.spines['left'].set_position(('data', 0))
    plt.title('tanh ')
    plt.plot(x, fx)
    plt.show()

if __name__ == "__main__":
    plot_sigmoid()
    plot_tanh()
    plot_relu()
    plot_softmax()