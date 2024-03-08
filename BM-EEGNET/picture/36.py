import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(1,4+1,1)
y = np.arange(1,4+1,1)
hist = (np.random.randint(0, 1000, 16)).reshape((4,4)) # 生成16个随机整数

zpos = 0
color = ('r','g','b','y')


# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.8
for i in range(4):
    c = color[i]
    ax.bar3d(range(4), [i] * 4, [0] * 4,
             dx, dy, hist[i, :],
             color=c)

# 设置坐标轴的刻度
ax.set_xticks(x)
ax.set_xlabel('X')

ax.set_yticks(y)
ax.set_ylabel('Y')

ax.set_zlabel('Z')

ax.view_init(elev=30,azim=-60)
# 将三维的灰色背诵面换成白色
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.show()


