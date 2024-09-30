import cv2
import numpy as np
import pytesseract

def preprocess_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro de ruído (Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar equalização de histograma para melhorar o contraste
    equalized = cv2.equalizeHist(blurred)
    
    # Aplicar a binarização (Thresholding)
    _, thresholded = cv2.threshold(equalized, 120, 255, cv2.THRESH_BINARY)
    
    return thresholded

def detect_plate(image):
    # Detectar contornos na imagem binarizada
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Definir um limite de área para descartar contornos muito pequenos ou muito grandes
    plate_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # O valor pode ser ajustado conforme o dataset
            # Aproximar o contorno em uma forma retangular
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Se tiver 4 vértices, pode ser uma placa
                plate_contour = approx
                break

    return plate_contour
def recognize_plate(image, plate_contour, original_image):
    if plate_contour is None:
        return "Placa não detectada"
    
    # Extrair a região da placa usando a máscara
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [plate_contour], -1, 255, -1)
    
    # Aplicar a máscara na imagem original
    plate = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Cortar a área onde está a placa
    x, y, w, h = cv2.boundingRect(plate_contour)
    cropped_plate = plate[y:y+h, x:x+w]
    
    # Converter a área da placa para escala de cinza e usar OCR
    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_plate, config='--psm 8')
    
    return text.strip()


# Usando o sistema completo
if __name__ == "__main__":
    placas = [
        "../images/image-1.jpg",
        "../images/image-2.jpg",
        "../images/image-3.jpg",
        "../images/image-4.jpg",
    ]
    
    for image_path in placas:
        try:
            # Pré-processar a imagem
            processed_image = preprocess_image(image_path)
            
            # Detectar a placa
            original_image = cv2.imread(image_path)
            plate_contour = detect_plate(processed_image)
            
            # Reconhecer os caracteres da placa
            plate_number = recognize_plate(processed_image, plate_contour, original_image)
            
            print("Placa Detectada:", plate_number)
        except Exception as e:
            print("Erro ao processar a imagem:", e)
            