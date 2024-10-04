import cv2
import numpy as np
import imutils

def adjust_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(img, table)

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def adjust_contrast(img):
    equalized = cv2.equalizeHist(img)
    clahe_img = clahe(equalized)
    gamma_img = adjust_gamma(clahe_img, 1.5)
    return gamma_img

def reduce_noise(img):
    return cv2.bilateralFilter(img, 13, 15, 15)

def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    return cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

def pre_processing_image(img):
    gray_img = adjust_color(img)
    contrast_img = adjust_contrast(gray_img)
    noise_img = reduce_noise(contrast_img)
    # sobel_img = sobel(noise_img)
    # cv2.imwrite('black_hat_img.png', sobel_img)
    return noise_img
