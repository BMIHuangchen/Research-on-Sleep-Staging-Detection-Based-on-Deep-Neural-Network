import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

N = 10
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)
# left表示从哪开始，
# radii表示从中心点向边缘绘制的长度（半径）
# width表示末端的弧长

# 自定义颜色和不透明度
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()