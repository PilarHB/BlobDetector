# Python code for Multiple Color Detection
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import canny


# Get image
image_path = './blob_images/blob_image.png'
image = cv2.imread(image_path)
cv2.imshow("Image", image)
cv2.waitKey()
cv2.destroyAllWindows()

#Convert to gray-scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey()
cv2.destroyAllWindows()
# convert the grayscale image to binary image
blurred = cv2.bilateralFilter(gray, 9,75, 75)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# ret, thresh = cv2.threshold(gray, 127, 255, 0)


# find contours in the binary image
contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)
    print(M)
    if M["m00"] != 0.0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    elif M["m00"] == 0.0:
        cX = 0
        cY = 0
    print(cX, cY)
    cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()


