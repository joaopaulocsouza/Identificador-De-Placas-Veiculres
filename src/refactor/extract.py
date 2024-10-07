import cv2
import numpy as np
import pytesseract

def extract_text(img):
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    text = pytesseract.image_to_string(img, lang='eng', config=config)
    print(f"Placa reconhecida: {text.strip()}")
    
    return pytesseract.image_to_string(img)