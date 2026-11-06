import cv2
import numpy as np

from A1_utils import load_gray, to_uint8, result_path

IMAGE_FILES = ['lenna.png', 'shapes.png']
KERNEL_SIZES = [5, 11, 17]
SIGMAS = [1, 6, 11]


# pad the image by copying the border pixels
def pad_replicate(img, pad_h, pad_w):
    h, w = img.shape
    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w))
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = img

    # top and bottom
    for i in range(pad_h):
        padded[i, pad_w:pad_w + w] = img[0, :]
        padded[pad_h + h + i, pad_w:pad_w + w] = img[h - 1, :]

    # left and right (this also fills the corners)
    for j in range(pad_w):
        padded[:, j] = padded[:, pad_w]
        padded[:, pad_w + w + j] = padded[:, pad_w + w - 1]

    return padded


# 1d cross correlation, output is the same size as img
def cross_correlation_1d(img, kernel):
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    h, w = img.shape
    out = np.zeros((h, w))

    if kernel.ndim == 1 or kernel.shape[0] == 1:
        # horizontal kernel
        k = kernel.flatten()
        padded = pad_replicate(img, 0, len(k) // 2)
        for j in range(len(k)):
            out += k[j] * padded[:, j:j + w]
    elif kernel.shape[1] == 1:
        # vertical kernel
        k = kernel.flatten()
        padded = pad_replicate(img, len(k) // 2, 0)
        for i in range(len(k)):
            out += k[i] * padded[i:i + h, :]
    else:
        raise ValueError('kernel should be 1d')

    return out


# 2d cross correlation, output is the same size as img
def cross_correlation_2d(img, kernel):
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    h, w = img.shape
    padded = pad_replicate(img, kh // 2, kw // 2)

    out = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + h, j:j + w]

    return out


# 1d gaussian, shape (1, size), sums to 1
def get_gaussian_filter_1d(size, sigma):
    half = size // 2
    kernel = np.zeros((1, size))
    for i in range(size):
        x = i - half
        kernel[0, i] = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    return kernel


# 2d gaussian, shape (size, size)
def get_gaussian_filter_2d(size, sigma):
    g = get_gaussian_filter_1d(size, sigma)
    return np.dot(g.T, g)


# vertical 1d filter first, then horizontal
def gaussian_filter_1d_sequential(img, size, sigma):
    g = get_gaussian_filter_1d(size, sigma)
    vertical = cross_correlation_1d(img, g.T)
    return cross_correlation_1d(vertical, g)


def gaussian_filter_2d(img, size, sigma):
    g = get_gaussian_filter_2d(size, sigma)
    return cross_correlation_2d(img, g)


def main():
    for file_name in IMAGE_FILES:
        img = load_gray(file_name)

        # 3x3 grid: rows = kernel sizes, columns = sigmas
        rows = []
        for size in KERNEL_SIZES:
            row = []
            for sigma in SIGMAS:
                row.append(to_uint8(gaussian_filter_2d(img, size, sigma)))
            rows.append(np.hstack(row))
        grid = np.vstack(rows)

        cv2.imwrite(result_path('part_1_gaussian_filtered_', file_name), grid)


if __name__ == '__main__':
    main()
