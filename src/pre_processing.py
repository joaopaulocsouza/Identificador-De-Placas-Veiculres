import cv2
import numpy as np

def adjust_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def adjust_gamma(img, gamma=1.15): 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(img, table)

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)) 
    return clahe.apply(img)

def adjust_contrast(img):
    equalized = cv2.equalizeHist(img)  
    clahe_img = clahe(equalized)       
    gamma_img = adjust_gamma(clahe_img)  
    return gamma_img


def reduce_noise(img):
    return cv2.bilateralFilter(img, 11, 15, 15) 

def pre_processing_image(img):
    gray_img = adjust_color(img)
    # contrast_img = adjust_contrast(gray_img)
    noise_img = reduce_noise(gray_img)
    return noise_img
