import cv2
import matplotlib.pyplot  as plt


img1 = cv2.imread('D:\project_python\BM-EEGNET\picture\picture\EEG.png',
                  cv2.IMREAD_COLOR)
img2 = cv2.imread('D:\project_python\BM-EEGNET\picture\picture\EEGGFP.png',
                  cv2.IMREAD_COLOR)
img3 = cv2.imread('D:\project_python\BM-EEGNET\picture\picture\EEGMean.png',
                  cv2.IMREAD_COLOR)
img4 = cv2.imread('D:\project_python\BM-EEGNET\picture\picture\EEGMedian.png',
                  cv2.IMREAD_COLOR)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)


plt.subplot(2, 2, 1)
plt.title("EEG信号")
plt.imshow(img1)

plt.subplot(2, 2, 2)
plt.title("基于GFP协议的EEG信号")
plt.imshow(img2)

plt.subplot(2, 2, 3)
plt.title("EEG信号平均值")
plt.imshow(img3)

plt.subplot(2, 2, 4)
plt.title("EEG信号中位数")
plt.imshow(img4)

plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.5,hspace=0.5)

plt.savefig(fname="D:\project_python\BM-EEGNET\picture\picture\EEGPictureMerge.png",dpi=300,format="png",)

plt.show()