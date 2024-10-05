import cv2
import numpy as np

def adjust_color(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def adjust_gamma(img, gamma=1.3):  # Gamma ajustado para 1.3
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    return cv2.LUT(img, table)

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE com grid 8x8
    return clahe.apply(img)

def adjust_contrast(img):
    equalized = cv2.equalizeHist(img)  # Equalização do histograma
    clahe_img = clahe(equalized)       # Aplicação de CLAHE
    gamma_img = adjust_gamma(clahe_img, 1.3)  # Ajuste de gamma
    return gamma_img

def reduce_noise(img):
    return cv2.bilateralFilter(img, 11, 15, 15)  # Filtro bilateral para suavização

def pre_processing_image(img):
    gray_img = adjust_color(img)
    # contrast_img = adjust_contrast(gray_img)
    noise_img = reduce_noise(gray_img)
    return noise_img
