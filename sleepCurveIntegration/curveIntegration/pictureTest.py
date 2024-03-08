import cv2
import matplotlib.pyplot  as plt

img1 = cv2.imread('D:/project_python/sleepCurveIntegration/curveIntegration/picture/Alpha_second_spectrum1.png',
                  cv2.IMREAD_COLOR)
img2 = cv2.imread('D:/project_python/sleepCurveIntegration/curveIntegration/picture/Alpha_second_spectrum2.png',
                  cv2.IMREAD_COLOR)
img3 = cv2.imread('D:/project_python/sleepCurveIntegration/curveIntegration/picture/Beta_second_spectrum1.png',
                  cv2.IMREAD_COLOR)
img4 = cv2.imread('D:/project_python/sleepCurveIntegration/curveIntegration/picture/Beta_second_spectrum2.png',
                  cv2.IMREAD_COLOR)
img5 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Delta_second_spectrum1.png',
                  cv2.IMREAD_COLOR)
img6 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Delta_second_spectrum2.png',
                  cv2.IMREAD_COLOR)
img7 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Gamma_second_spectrum1.png',
                  cv2.IMREAD_COLOR)
img8 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Gamma_second_spectrum2.png',
                  cv2.IMREAD_COLOR)
img9 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Theta_second_spectrum1.png',
                  cv2.IMREAD_COLOR)
img10 = cv2.imread('D:/project_python/sleepCurveIntegration//curveIntegration/picture/Theta_second_spectrum2.png',
                   cv2.IMREAD_COLOR)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)
img6 = cv2.cvtColor(img6, cv2.COLOR_BGR2RGB)
img7 = cv2.cvtColor(img7, cv2.COLOR_BGR2RGB)
img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
img9 = cv2.cvtColor(img9, cv2.COLOR_BGR2RGB)
img10 = cv2.cvtColor(img10, cv2.COLOR_BGR2RGB)




plt.subplot(2, 5, 1)
plt.title("(a)Alpha")
plt.imshow(img1)

plt.subplot(2, 5, 6)
plt.title("(a)Alpha")
plt.imshow(img2)

plt.subplot(2, 5, 2)
plt.title("(b)Beta")
plt.imshow(img3)

plt.subplot(2, 5, 7)
plt.title("(b)Beta")
plt.imshow(img4)

plt.subplot(2, 5, 3)
plt.title("(c)Delta")
plt.imshow(img5)

plt.subplot(2, 5, 8)
plt.title("(c)Delta")
plt.imshow(img6)

plt.subplot(2, 5, 4)
plt.title("(d)Gamma")
plt.imshow(img7)

plt.subplot(2, 5, 9)
plt.title("(d)Gamma")
plt.imshow(img8)

plt.subplot(2, 5, 5)
plt.title("(e)Theta")
plt.imshow(img9)

plt.subplot(2, 5, 10)
plt.title("(e)Theta")
plt.imshow(img10)


plt.show()
