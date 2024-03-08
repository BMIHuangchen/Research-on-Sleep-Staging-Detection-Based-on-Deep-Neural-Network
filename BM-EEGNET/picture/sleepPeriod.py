import matplotlib.pyplot as plt
import numpy as np


def func(x):
    return -(x - 2) * (x - 8) + 40

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x = np.arange(0, 6, 0.1)
y = np.sinc(x)
x1 = x[7:]
a = y[7:]*10
plt.plot(x1, a, linewidth=2)
plt.xticks([])
plt.yticks([])
plt.text(4, 3.5, s="NREM--非快速眼动期",fontsize=14, style='oblique', ha='left', va='top', wrap=True)
plt.text(4, 3, s="REM--快速眼动期", fontsize=14, style='oblique',ha='left', va='top', wrap=True)
props = dict(facecolor='black')
plt.annotate('NREM', xytext=(1.45, 1), xy=(1.45, -1), arrowprops=props, fontsize=14, ha="center")
plt.annotate('REM', xytext=(2.45, -1), xy=(2.45, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.annotate('NREM', xytext=(3.45, 1), xy=(3.45, -0.5), arrowprops=props, fontsize=14, ha="center")
plt.annotate('REM', xytext=(4.5, -1), xy=(4.5, 0.5), arrowprops=props, fontsize=14, ha="center")
plt.annotate('NREM', xytext=(5.5, 1), xy=(5.5, -0.3), arrowprops=props, fontsize=14, ha="center")
plt.title("sleepPeriod", fontsize=14)



plt.show()
