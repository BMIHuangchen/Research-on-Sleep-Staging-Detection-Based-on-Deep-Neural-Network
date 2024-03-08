import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):  #定义himmelblau函数
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6,6,0.1) #设置x范围
y = np.arange(-6,6,0.1) #设置y范围
print('x,y range:',x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print('x,y maps:',x.shape,y.shape)
Z = himmelblau([X,Y])

fig = plt.figure("himmelblau")
Axes3D = fig.gca(projection='3d')
Axes3D.plot_surface(X,Y,Z)
Axes3D.view_init(60,-30)
Axes3D.set_xlabel('x')
Axes3D.set_ylabel('y')
plt.show()
