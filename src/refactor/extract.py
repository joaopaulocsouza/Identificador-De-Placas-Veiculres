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
    return cv2.Canny(img, 100, 200)

def erode(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def draw(img, plate):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [plate], 0, 255, -1)
    (x, y, w, h) = cv2.boundingRect(plate)
    cropped = img[y:y+h, x:x+w]
    return cropped

def fill_holes(img):
    # Inverter a imagem para que as áreas escuras se tornem claras
    inverted_img = cv2.bitwise_not(img)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(inverted_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Preencher buracos
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
        cropped = draw(inverted_img, approx)
        cv2.imshow('Cropped', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Reverter a inversão para manter a imagem final no formato original
    filled_img = cv2.bitwise_not(inverted_img)
    
    return filled_img

def apply_filters(img):
    canny_img = canny(img)
    filled_img = fill_holes(canny_img)
    return canny_img


def extract_text(img, original_plate):
    filtered_img = apply_filters(img)
    cv2.imshow('Filtered', apply_filters(img))
    cv2.imshow('original', apply_filters(original_plate))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    text = pytesseract.image_to_string(img, lang='eng', config=config)
    print(f"Placa reconhecida: {text.strip()}")
    
    return pytesseract.image_to_string(img)