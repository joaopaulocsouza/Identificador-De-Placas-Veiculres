import cv2
import numpy as np
import imutils
import os

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


def preprocessing(path):
    #Lendo imagem
    img=cv2.imread(path)
    #Convertendo para GrayScale
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Reduzindo ruídos
    blur=cv2.bilateralFilter(gray, 9,75,75)
#    cv2.imshow('image',blur)
#    cv2.waitKey(0)
    #Operação morfologica Black-hat
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (10,3))
    black_hat=cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
#    cv2.imshow('image',black_hat)
#    cv2.waitKey(0)
    #Operação de fechamento
    kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    close = cv2.morphologyEx(black_hat, cv2.MORPH_CLOSE, kernel2)
#   cv2.imshow('image',close)
#    cv2.waitKey(0)
    #Binarização
    thresh= cv2.threshold(close,0 , 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #Gradiente na Direção x
    gradient_x = cv2.Sobel(thresh, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    gradient_x=normalizar(gradient_x)
    #cv2.imshow('image',gradient_x)
    #cv2.waitKey(0)
    #Redução de ruidos
    blur=cv2.GaussianBlur(gradient_x, (9,9), 0)
    #Operação de Fechamento
    close2=cv2.morphologyEx(blur,cv2.MORPH_CLOSE,kernel2)
    thresh2=cv2.threshold(close2,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #Erosão+Dilatação
    erode=cv2.erode(thresh2,None, iterations=3)#2
    dilate=cv2.dilate(erode, None, iterations=9)#1
    cv2.imshow('image', dilate)
    cv2.waitKey(0)
    croped=crop(dilate, gray)
    if croped is not None:
        cv2.imshow('image', croped)
        cv2.waitKey(0)
        cv2.imwrite('croped/crop01.jpg', croped)
        return 'croped/crop01.jpg'
    else:
        print('Placa não encontrada!')
        return None

def crop(img, gray):
    contornos= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont= imutils.grab_contours(contornos)
    cnts= sorted(cont, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        (x,y,w,h)=cv2.boundingRect(c)
        p=w/h
        #print(p)
        if p>= 2 and p<=5:
            crop= gray[y:y+h, x:x+w]
            return crop
        
# def detect_and_recognize_plate(image_path):
#     if not os.path.isfile(image_path):
#         print(f"Erro: arquivo {image_path} não encontrado.")
#         return

#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Erro ao carregar a imagem: {image_path}")
#         return

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     edged = cv2.Canny(gray, 30, 200)
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

#     plate = None
#     for c in contours:
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.018 * peri, True)

#         if len(approx) == 4:
#             plate = approx
#             break

#     if plate is None:
#         print("Placa não encontrada")
#         return

#     mask = np.zeros(gray.shape, np.uint8)
#     cv2.drawContours(mask, [plate], -1, 255, -1)

#     (x, y, w, h) = cv2.boundingRect(plate)
#     cropped = gray[y:y+h, x:x+w]

#     text = pytesseract.image_to_string(cropped, config='--psm 8')
#     print(f"Placa reconhecida: {text.strip()}")

#     cv2.imwrite("placa_recortada.jpg", cropped)
#     print(f"Imagem recortada da placa salva como: placa_recortada.jpg")
#     print(f"Texto reconhecido: {text.strip()}")

def main():
    extensions = ['.png', '.jpg', '.jpeg']
    
    for file in os.listdir('./images'):
        path = os.path.join('./images', file)
        print(file)
        
        if os.path.isfile(path) and os.path.splitext(path)[1] in extensions:  
            preprocessing(path)
            
            # cv2.imshow('Imagem pre processed', img)
            # cv2.imshow('Plate', img_pre_processed)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
main()