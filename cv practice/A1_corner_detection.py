import cv2
import numpy as np

from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
from A1_utils import SOBEL_X, SOBEL_Y, load_gray, to_uint8, result_path

KAPPA = 0.04
WINDOW = 5  # window size for the second moment matrix
THRESHOLD = 0.1
NMS_WIN = 11
GREEN = (0, 255, 0)  # BGR
IMAGE_FILES = ['shapes.png', 'lenna.png']


# harris response, normalized to 0 ~ 1
def compute_corner_response(img):
    ix = cross_correlation_2d(img, SOBEL_X)
    iy = cross_correlation_2d(img, SOBEL_Y)

    # summing over the 5x5 window = correlation with a 5x5 kernel of ones
    window = np.ones((WINDOW, WINDOW))
    sxx = cross_correlation_2d(ix * ix, window)
    syy = cross_correlation_2d(iy * iy, window)
    sxy = cross_correlation_2d(ix * iy, window)

    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    R = det - KAPPA * trace ** 2

    # edges give negative values -> set to 0
    R[R < 0] = 0

    # normalize
    if R.max() > 0:
        R = R / R.max()
    return R


# keep only the pixels that are the max inside a winSize x winSize window
def non_maximum_suppression_win(R, winSize):
    h, w = R.shape
    half = winSize // 2
    result = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            # skip anything under the threshold
            if R[y, x] <= THRESHOLD:
                continue

            # window around (y, x), cut off at the image border
            y1 = max(0, y - half)
            y2 = min(h, y - half + winSize)
            x1 = max(0, x - half)
            x2 = min(w, x - half + winSize)

            if R[y, x] >= R[y1:y2, x1:x2].max():
                result[y, x] = R[y, x]

    return result


def main():
    gaussian = get_gaussian_filter_2d(7, 1.5)

    for file_name in IMAGE_FILES:
        img = load_gray(file_name)
        smoothed = cross_correlation_2d(img, gaussian)

        # corner response
        R = compute_corner_response(smoothed)
        cv2.imwrite(result_path('part_3_corner_raw_', file_name), to_uint8(R * 255.0))

        # threshold -> green pixels on the original image
        color = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
        color[R > THRESHOLD] = GREEN
        cv2.imwrite(result_path('part_3_corner_bin_', file_name), color)

        # nms -> green circles on the original image
        sup = non_maximum_suppression_win(R, NMS_WIN)
        ys, xs = np.nonzero(sup)
        circles = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
        for i in range(len(xs)):
            cv2.circle(circles, (int(xs[i]), int(ys[i])), 5, GREEN, 2)
        cv2.imwrite(result_path('part_3_corner_sup_', file_name), circles)


if __name__ == '__main__':
    main()
