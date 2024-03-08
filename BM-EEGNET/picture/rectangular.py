import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()-0.2, 1.01*height, '%s' % float(height),size=13)
shops = ["Wake", "N1", "N2", "N3", "REM"]
w=[88.1,90.4,89.2]
n1=[45.6,52.9,55.4]
n2=[88.5,87.4,88.0]
n3=[89.9,86.4,88.1]
rem=[83.6,80.8,82.2]
sales_product_1 = [w[0], n1[0], n2[0], n3[0], rem[0]]
sales_product_2 = [w[1], n1[1], n2[1], n3[1], rem[1]]
sales_product_3 = [w[2], n1[2], n2[2], n3[2], rem[2]]
# sales_product_1 = [100, 85, 56, 42, 72, 15]
# sales_product_2 = [50, 120, 65, 85, 25, 55]
# sales_product_3 = [20, 35, 45, 27, 55, 65]
font1 = {'size': 23, 'weight': 'normal', }
# 创建分组柱状图，需要自己控制x轴坐标
xticks = np.arange(len(shops))

fig, ax = plt.subplots(figsize=(10, 5))
# 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
a= ax.bar(xticks, sales_product_1, width=0.25, label="准确率", color="red")
# 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
b= ax.bar(xticks + 0.25, sales_product_2, width=0.25, label="灵敏度", color="yellow")
# 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
c= ax.bar(xticks + 0.5, sales_product_3, width=0.25, label="F1值", color="blue")

autolabel(a)
autolabel(b)
autolabel(c)

ax.set_title("睡眠各阶段指标",font1)
ax.set_xlabel("sleep stage",font1)
ax.set_ylabel("score/%",font1)
ax.legend( loc='lower right',fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 最后调整x轴标签的位置
ax.set_xticks(xticks + 0.25)
ax.set_xticklabels(shops)
plt.show()
