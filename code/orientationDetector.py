import numpy as np
import cv2
import matplotlib.pyplot as plt


def getOrientation(img):
    image = cv2.imread(img)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im_size = image.shape
    img2 = image[int((im_size[0] / 2) - 150):int((im_size[0] / 2) + 185),
           int((im_size[1] / 2)) - 195:int((im_size[1] / 2) + 140), :]
    img2Colors = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, bw = cv2.threshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cntrs = contours[0] if len(contours) == 2 else contours[1]
    for i, c in enumerate(contours):

        # cv.minAreaRect returns:
        # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Retrieve the key parameters of the rotated bounding box
        center = (int(rect[0][0]), int(rect[0][1]))
        width = int(rect[1][0])
        height = int(rect[1][1])
        angle = int(rect[2])

        if width < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    print(angle, "deg")
    plt.imshow(img2Colors)
    plt.title("angle: "+ str(angle)+ " deg")
    plt.show()
    return angle

