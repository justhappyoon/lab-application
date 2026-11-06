import cv2
import numpy as np

from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
from A1_utils import SOBEL_X, SOBEL_Y, load_gray, to_uint8, result_path

IMAGE_FILES = ['shapes.png', 'lenna.png']

# row / col steps for 0, 45, 90, ... 315 degrees
# (row goes up (-1) for the upper directions because iy points up)
DY = [0, -1, -1, -1, 0, 1, 1, 1]
DX = [1, 1, 0, -1, -1, -1, 0, 1]


# returns magnitude and direction (in radians) using sobel
def compute_image_gradient(img):
    ix = cross_correlation_2d(img, SOBEL_X)
    iy = cross_correlation_2d(img, SOBEL_Y)
    mag = np.sqrt(ix ** 2 + iy ** 2)
    direction = np.arctan2(iy, ix)
    return mag, direction


# keep a pixel only if it is bigger than both neighbors along its gradient direction
def non_maximum_suppression_dir(mag, dir):
    h, w = mag.shape
    result = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            # angle in degrees (0 ~ 360), then snap to the closest multiple of 45
            angle = np.degrees(dir[y, x]) % 360
            b = int(round(angle / 45)) % 8

            # the two neighbors on the gradient line
            y1 = y + DY[b]
            x1 = x + DX[b]
            y2 = y - DY[b]
            x2 = x - DX[b]

            # outside the image counts as 0
            n1 = 0
            n2 = 0
            if y1 >= 0 and y1 < h and x1 >= 0 and x1 < w:
                n1 = mag[y1, x1]
            if y2 >= 0 and y2 < h and x2 >= 0 and x2 < w:
                n2 = mag[y2, x2]

            if mag[y, x] > n1 and mag[y, x] > n2:
                result[y, x] = mag[y, x]

    return result


def main():
    gaussian = get_gaussian_filter_2d(7, 1.5)

    for file_name in IMAGE_FILES:
        img = load_gray(file_name)
        smoothed = cross_correlation_2d(img, gaussian)

        # raw edges
        mag, direction = compute_image_gradient(smoothed)
        cv2.imwrite(result_path('part_2_edge_raw_', file_name), to_uint8(mag))

        # suppressed edges
        sup = non_maximum_suppression_dir(mag, direction)
        cv2.imwrite(result_path('part_2_edge_sup_', file_name), to_uint8(sup))


if __name__ == '__main__':
    main()
