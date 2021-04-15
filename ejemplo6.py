import numpy as np
import cv2
 
original = cv2.imread("monedas.png")
cv2.imshow("original", original)
cv2.waitKey(0)

gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("gris", gris)
cv2.waitKey(0) 

gauss = cv2.GaussianBlur(gris, (15,15), 0)
cv2.imshow("suavizado", gauss)
cv2.waitKey(0)

canny = cv2.Canny(gauss, 50, 150)
cv2.imshow("canny", canny)
cv2.waitKey(0)

(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Número de contornos {} ".format(len(contornos)))

cv2.drawContours(original,contornos,-1,(0,0,255), 2)
cv2.imshow("Contornos", original)
cv2.waitKey(0)

cv2.destroyAllWindows()