# Python code for Blob Detection
import imutils
import numpy as np
import cv2
from camera_calibration.perspective_calibration import PerspectiveCalibration
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import canny


class BlobDetector:
    def __init__(self, x_length, y_length):
        # Setup camera with camera calibration
        self.pc = PerspectiveCalibration()
        self.pc.setup_camera()
        # Get image
        self.image_path = '../blob_images/img1610361325.5.png'
        self.image = cv2.imread(self.image_path)
        self.quadrants = self.get_quadrants(x_length, y_length)

    # TODO: Calculate border of our box from real environment to pixels
    def get_quadrants(self, x_length, y_length):
        x_length = x_length / 0.054
        y_length = y_length / 0.054

        # Corners
        upper_right_corner = (-x_length / 2, y_length / 2, 0)
        upper_left_corner = (-x_length / 2, -y_length / 2, 0)
        lower_right_corner = (x_length / 2, y_length / 2, 0)
        lower_left_corner = (x_length / 2, -y_length / 2, 0)

        # Center
        center = (0.0, 0.0, 0.0)
        center_right_up = (-x_length / 6, y_length / 6, 0)
        center_right_down = (x_length / 6, y_length / 6, 0)
        center_left_down = (x_length / 6, -y_length / 6, 0)
        center_left_up = (-x_length / 6, -y_length / 6, 0)

        # Middles
        middle_right_up = (-x_length / 6, y_length / 2, 0)
        middle_right_down = (x_length / 6, y_length / 2, 0)
        middle_left_up = (-x_length / 6, -y_length / 2, 0)
        middle_left_down = (x_length / 6, -y_length / 2, 0)
        middle_down_left = (x_length / 2, -y_length / 6, 0)
        middle_down_right = (x_length / 2, y_length / 6, 0)
        middle_up_left = (-x_length / 2, -y_length / 6, 0)
        middle_up_right = (x_length / 2, y_length / 6, 0)

        upper_left_corner_pixels = self.pc.from_3d_to_2d(self.image, upper_left_corner, draw=True)[0][0]
        upper_right_corner_pixels = self.pc.from_3d_to_2d(self.image, upper_right_corner, draw=True)[0][0]
        lower_left_corner_pixels = self.pc.from_3d_to_2d(self.image, lower_left_corner, draw=True)[0][0]
        lower_right_corner_pixels = self.pc.from_3d_to_2d(self.image, lower_right_corner, draw=True)[0][0]

        center_pixels = self.pc.from_3d_to_2d(self.image, center, draw=True)[0][0]
        # middle_right_pixels = from_3d_to_2d(self.image, middle_right, draw=True)[0][0]
        # middle_left_pixels = from_3d_to_2d(self.image, middle_left, draw=True)[0][0]
        # middle_down_pixels = from_3d_to_2d(self.image, middle_down, draw=True)[0][0]
        # middle_up_pixels = from_3d_to_2d(self.image, middle_up, draw=True)[0][0]

        middle_right_up_pixels = self.pc.from_3d_to_2d(self.image, middle_right_up, draw=True)[0][0]
        middle_right_down_pixels = self.pc.from_3d_to_2d(self.image, middle_right_down, draw=True)[0][0]
        middle_left_up_pixels = self.pc.from_3d_to_2d(self.image, middle_left_up, draw=True)[0][0]
        middle_left_down_pixels = self.pc.from_3d_to_2d(self.image, middle_left_down, draw=True)[0][0]
        middle_down_left_pixels = self.pc.from_3d_to_2d(self.image, middle_down_left, draw=True)[0][0]
        middle_down_right_pixels = self.pc.from_3d_to_2d(self.image, middle_down_right, draw=True)[0][0]
        middle_up_left_pixels = self.pc.from_3d_to_2d(self.image, middle_up_left, draw=True)[0][0]
        middle_up_right_pixels = self.pc.from_3d_to_2d(self.image, middle_up_right, draw=True)[0][0]
        center_down_left_pixels = self.pc.from_3d_to_2d(self.image, center_left_down, draw=True)[0][0]
        center_down_right_pixels = self.pc.from_3d_to_2d(self.image, center_right_down, draw=True)[0][0]
        center_up_left_pixels = self.pc.from_3d_to_2d(self.image, center_left_up, draw=True)[0][0]
        center_up_right_pixels = self.pc.from_3d_to_2d(self.image, center_right_up, draw=True)[0][0]

        quadrants = [
            [upper_left_corner_pixels, middle_up_left_pixels, center_up_left_pixels, middle_left_up_pixels],
            [middle_up_left_pixels, middle_up_right_pixels, center_up_right_pixels, center_up_left_pixels],
            [middle_up_right_pixels, upper_right_corner_pixels, middle_right_up_pixels, center_up_right_pixels],
            [middle_left_up_pixels, center_up_left_pixels, center_down_left_pixels, middle_left_down_pixels],
            [center_up_left_pixels, center_up_right_pixels, center_down_right_pixels, center_down_left_pixels],
            [center_up_right_pixels, middle_right_up_pixels, middle_right_down_pixels, center_down_right_pixels],
            [middle_left_down_pixels, center_down_left_pixels, middle_down_left_pixels, lower_left_corner],
            [center_down_left_pixels, center_down_right_pixels, middle_down_right_pixels, middle_down_left_pixels],
            [center_down_right_pixels, middle_right_down_pixels, lower_right_corner_pixels, middle_down_right_pixels]
        ]
        return quadrants

    def check_pixel_quadrant(self, cx, cy):
        def limit(a1, a2, a, b1, b2):
            return (((b2 - b1) / (a2 - a1)) * (a - a1)) + b1

        for idx, quadrant in enumerate(self.quadrants):
            check_1 = cy >= limit(quadrant[0][0], quadrant[1][0], cx, quadrant[0][1], quadrant[1][1])
            check_2 = cx >= limit(quadrant[0][1], quadrant[3][1], cy, quadrant[0][0], quadrant[3][0])
            check_3 = cx < limit(quadrant[1][1], quadrant[2][1], cy, quadrant[1][0], quadrant[2][0])
            check_4 = cy < limit(quadrant[3][0], quadrant[2][0], cx, quadrant[3][1], quadrant[2][1])
            if check_1 and check_2 and check_3 and check_4:
                return idx
        return None

    @staticmethod
    def _find_contours(thresh):
        # Find contours in the binary image
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        return contours

    def find_optimal_quadrant(self, image):
        thresh = self._image_preprocessing(image)
        contours = self._find_contours(thresh)
        # initialize quadrant count
        index_quadrant = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            if M["m00"] != 0.0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                index = self.check_pixel_quadrant(cX, cY)
                if index is not None:
                    index_quadrant[index] += 1
            else:
                cX = 0
                cY = 0

        # display the image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return index_quadrant.index(max(index_quadrant))

    @staticmethod
    def _image_preprocessing(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert to gray-scale
        # Convert the grayscale image to binary image
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return thresh


if __name__ == '__main__':
    image_path = '../blob_images/img1610361325.5.png'
    image = cv2.imread(image_path)
    blob_detector = BlobDetector(x_length=0.13, y_length=0.19)
    print(blob_detector.find_optimal_quadrant(image))
