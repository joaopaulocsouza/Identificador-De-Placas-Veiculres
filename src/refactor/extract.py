import cv2
import numpy as np
import pytesseract

def normalizar(gradiente):
    '''
    Normaliza os valores do gradiente da imagem binarizada
    :param gradiente: Imagem binarizada após aplicação do cv2.Sobel() na direção x
    :return: Gradiente normalizado em uint8 com valores entre [0-255]
    '''
    gradiente=np.absolute(gradiente)
    (min, max)=np.min(gradiente), np.max(gradiente)
    gradiente=255*((gradiente-min)/(max-min))
    gradiente=gradiente.astype('uint8')
    return gradiente



def black_hat_transform(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

def closing_transform(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def threshold_transform(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def canny(img):
    return cv2.Canny(img, 30, 200)

def erode(img):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def dilate(img):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def fill_holes(img):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

def apply_filters(img):
    black_hat_img = black_hat_transform(img)
    closing_transform_img = closing_transform(black_hat_img)
    threshold_img = threshold_transform(closing_transform_img)
    canny_img = canny(img)
    erode_img = erode(canny_img)
    dilated_img = dilate(canny_img)
    combined = cv2.subtract(dilated_img, erode_img)
    filled_img = fill_holes(combined)
    cv2.imshow('Filled', filled_img)
    
    return filled_img


def extract_text(img):
    filtered_img = apply_filters(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    text = pytesseract.image_to_string(img, lang='eng', config=config)
    print(f"Placa reconhecida: {text.strip()}")
    
    return pytesseract.image_to_string(img)