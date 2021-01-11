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

# # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
# # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV-Image", image)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
#
#
#
#
# # Set range for white color and define mask
# white_lower = np.array([0, 0, 100], np.uint8)
# white_upper = np.array([0, 0, 255], np.uint8) # dark gray
# white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
#
#
# # Set range for brown color and define
# brown_lower = np.array([67, 18, 58], np.uint8)
# brown_upper = np.array([46, 37, 42], np.uint8)
# brown_mask = cv2.inRange(hsv_image, brown_upper, brown_lower)
#
# # Morphological Transform, Dilation for each color and bitwise_and operator between imageFrame and mask
# # determines to detect only that particular color
# kernal = np.ones((5, 5), "uint8")
#
# # # For white color
# white_mask = cv2.dilate(white_mask, kernal)
# cv2.imshow("Mask_white", white_mask)
# cv2.waitKey()
# cv2.destroyAllWindows()
# res_white = cv2.bitwise_and(hsv_image, hsv_image, mask=white_mask)
# cv2.imshow("Res_white", res_white)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# #
# # # For brown color
# brown_mask = cv2.dilate(brown_mask, kernal)
# cv2.imshow("Brown_Mask", brown_mask)
# cv2.waitKey()
# cv2.destroyAllWindows()
# res_brown = cv2.bitwise_and(hsv_image, hsv_image, mask=brown_mask)
# cv2.imshow("Brown_Res", brown_mask)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # Creating contour to track white color
# contours, hierarchy = cv2.findContours(white_mask,
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
#
# print(contours)
#
# for pic, contour in enumerate(contours):
#     area = cv2.contourArea(contour)
#     if area > 50:
#         x, y, w, h = cv2.boundingRect(contour)
#         imageFrame = cv2.rectangle(hsv_image, (x, y),
#                                    (x + w, y + h),
#                                    (0, 0, 255), 2)
#
#         cv2.putText(imageFrame, "White", (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#                     (0, 0, 255))
#
#         cv2.imshow("White-Detector", imageFrame)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
#     else:
#         print("else")
