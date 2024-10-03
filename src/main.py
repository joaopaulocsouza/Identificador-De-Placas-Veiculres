import cv2
import numpy as np
import pytesseract
import pre_processing
import detection

images = [
    '../images/image-1.png', 
    '../images/image-2.png', 
    '../images/image-3.png',
    '../images/image-4.png', 
    '../images/image-5.jpg', 
    '../images/image-6.png', 
]

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    for image in images:
        img = cv2.imread(image)
        img_pre_processed = pre_processing.pre_processing_image(img)
        countours = detection.find_countours(img_pre_processed)
        plates = []
        for countour in countours:
            x, y, w, h = cv2.boundingRect(countour)
            proportion = w / float(h)
            
            if 2.0 < proportion < 5.0 and w > 50 and h > 20:
                plates.append((x, y, w, h))
            
        for plate in plates:
            x, y, w, h = plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = img[y:y + h, x:x + w]
            cv2.imshow('Placa', roi)
            cv2.waitKey(0)
            text = pytesseract.image_to_string(roi)
            if len(text) > 4:
                print(text ,len(text))
            # if text:
            #     print(text)
        # cv2.imshow('Imagem Original', img)
        cv2.imshow('Imagem após ajustes inicial', img_pre_processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# # Aplicar a convolução com o kernel passa-alta
#         blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#         A = 1.5  # A > 1 para filtro High-Boost

#         # Aplicar o filtro High-Boost: HighBoost = A * imagem original - imagem suavizada
#         high_boost = cv2.addWeighted(gray, A, blurred, -1, 0)
        # laplacian = cv2.Laplacian(high_boost, cv2.CV_64F)
        # sobelx = cv2.Sobel(high_boost, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(high_boost, cv2.CV_64F, 0, 1, ksize=3)
        # filtered_image_hb = cv2.filter2D(high_boost, -1, kernel)
        # filtered_image = cv2.filter2D(gray, -1, kernel)
        # cv2.imshow('Imagem com Filtro Passa-Alta', filtered_image)
        # cv2.imshow('Imagem com Filtro Passa-Alta HB', filtered_image_hb)
        # cv2.imshow('Imagem com Filtro laplacian', laplacian)
        # cv2.imshow('Imagem com Filtro sobely', sobely)
        # cv2.imshow('Imagem com Filtro sobelx', sobelx)
        # cv2.imshow('Filtro High-Boost', high_boost)
        # cv2.imshow('Gray', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # img = pre_processing.pre_processing_image(img)
        # show_image(img)
        # print(result)

main()