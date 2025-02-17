import cv2
import numpy as np
from extract import extract_text


def fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft) 
 
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2  

    r = 240  
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1  

    fshift = dft_shift * mask
  
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    img_back = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    return img_back

def apply_filters(img):
    cannied = cv2.Canny(img, 30, 200)
    # fourier_img = fourier_transform(cannied)
    return cannied

def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:
            return approx
        
    return None

def adjust_plate(plate):
    _, limiarized = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing_img = cv2.morphologyEx(limiarized, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    dilated_img = cv2.dilate(opening_img, kernel, iterations=1)
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
    
    cv2.imshow('Ajusted plate', eroded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return eroded_img

def draw(img, plate):
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [plate], 0, 255, -1)
    (x, y, w, h) = cv2.boundingRect(plate)
    cropped = img[y:y+h, x:x+w]
    return cropped


def detection(img, original, name):
    filtered_img = apply_filters(img)
    plate = find_contours(filtered_img)
    if plate is not None:
        cropped = draw(original, plate)
        cropped_filtered = draw(img, plate)
        ajusted_img = adjust_plate(cropped_filtered)
        extract_text(ajusted_img)
        cv2.imwrite(f"../out/plate/{name}_placa_recortada.jpg", cropped)
        
    return filtered_img