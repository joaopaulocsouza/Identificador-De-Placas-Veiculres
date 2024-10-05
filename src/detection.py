import cv2
import numpy as np
import pytesseract
import imutils

def canny(img):
    return cv2.Canny(img, 30, 200)

def dilate(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def erode(img):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def find_contours(img):
    contours = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            return approx
        
    return None


def detection(img_processed, img, name):
    canny_img = canny(img_processed)
    cv2.imshow('canny_detection', canny_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plate = find_contours(canny_img)
    
    if plate is None:
        print('No plate detected')
        return img
    
    else:
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [plate], 0, 255, -1)
        (x, y, w, h) = cv2.boundingRect(plate)
        cropped = img[y:y+h, x:x+w]
        
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        text = pytesseract.image_to_string(cropped, lang='eng', config=config)
        print(f"Placa reconhecida: {text.strip()}")
        
        cv2.imshow('Image', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite(f"./out/{name}_placa_recortada.jpg", cropped)
        print(f"Imagem recortada da placa salva como: {name}_placa_recortada.jpg")
        print(f"Texto reconhecido: {text.strip()}")
    
    return cropped