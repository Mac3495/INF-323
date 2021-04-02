import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gato.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

historigram = cv2.calcHist([img_gray], [0], None, [256], [0, 255])

fig = plt.figure(1)
pl = fig.add_subplot(311)
pl.plot(historigram)
fig.show()

cv2.imshow('Imagen', img)
cv2.imshow('Imagen GRIS', img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.close(fig)