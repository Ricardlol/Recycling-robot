import matplotlib.pyplot as plt
import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np

def getOrientation(img):
    #image = cv2.imread("./img/3bfce634-ab0c-4223-84c7-c20df6f6b48d.png")
    image = cv2.imread(img)

    backgound = cv2.imread("./background.png")

    imgResult = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_blur = cv2.GaussianBlur(image, (51, 51), cv2.BORDER_DEFAULT)
    image_bw = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)

    backblur = cv2.GaussianBlur(backgound, (51, 51), cv2.BORDER_DEFAULT)
    back_bw = cv2.cvtColor(backblur, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(image_bw, back_bw)
    ret, mascara = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel_1 = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara, kernel_1, iterations=5)
    mascara = cv2.dilate(mascara, kernel_1, iterations=2)

    kernel_2 = np.ones((7, 7), np.uint8)
    mascara = cv2.dilate(mascara, kernel_2, iterations=20)
    mascara = cv2.erode(mascara, kernel_2, iterations=10)

    contours = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    rotrect = cv2.minAreaRect(contours[0])

    cv2.drawContours(imgResult, contours, 0,(255,0,0),2)

    angle = rotrect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = angle

    print(angle, "deg")
    plt.imshow(imgResult)
    plt.title("angle: "+ str(angle)+ " deg")
    plt.show()
    return angle

