import cv2
import numpy as np

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def apply_filters(img):
    gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.bilateralFilter(gaussian, 9, 75, 75)

def limiarization(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def adjust_gamma(img, gamma=1.15): 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(img, table)

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)) 
    return clahe.apply(img)

def adjust_contrast(img):
    clahe_img = clahe(img)       
    gamma_img = adjust_gamma(clahe_img)  
    return gamma_img

    
def pre_processing_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histogram_img = histogram_equalization(gray_img)
    # contrasted_img = adjust_contrast(gray_img)
    filtered_img = apply_filters(gray_img) 
    return filtered_img