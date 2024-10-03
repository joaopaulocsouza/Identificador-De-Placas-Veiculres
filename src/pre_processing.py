import cv2
import numpy as np

def save_image(img):
    cv2.imwrite('placa.jpg', img)
    
    # Ajustes iniciais

def intial_adjustments(img):
    # Convertendo a imagem para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Equalização do histograma
    equalized = cv2.equalizeHist(gray)
    return equalized

def apply_high_boost_filter(img, A):
    # Aplicar a convolução com o kernel passa-alta
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    # Aplicar o filtro High-Boost: HighBoost = A * imagem original - imagem suavizada
    high_boost = cv2.addWeighted(img, A, blurred, -1, 0)
    # Criar um kernel passa-alta (por exemplo, para realçar bordas)
    kernel = np.array([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1]])

    filtered_image = cv2.filter2D(high_boost, -1, kernel)
    return filtered_image


# Função para aplicar a filtragem homomórfica
def homomorphic_filter(image, d0=30, rh=1.0, rl=0.5, c=0.9):
    # Passo 2: Aplicar logaritmo na imagem
    gray_log = np.log1p(np.array(image, dtype="float"))

    # Passo 3: Transformada de Fourier
    dft = np.fft.fft2(gray_log)
    dft_shift = np.fft.fftshift(dft)

    # Passo 4: Criar o filtro passa-alta homomórfico no domínio da frequência
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2  # Coordenadas do centro

    # Construir a máscara do filtro homomórfico
    H = np.zeros((rows, cols), dtype=np.float32)
    for u in range(rows):
        for v in range(cols):
            duv = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            H[u, v] = (rh - rl) * (1 - np.exp(-c * (duv ** 2) / (d0 ** 2))) + rl

    # Passo 5: Aplicar o filtro no domínio da frequência
    filtered_dft = dft_shift * H

    # Passo 6: Transformada inversa de Fourier
    idft_shift = np.fft.ifftshift(filtered_dft)
    idft = np.fft.ifft2(idft_shift)
    idft_exp = np.exp(np.abs(idft)) - 1  # Desfazendo o logaritmo

    # Normalizar a imagem para o intervalo de 0-255
    idft_exp = np.uint8(cv2.normalize(idft_exp, None, 0, 255, cv2.NORM_MINMAX))

    return idft_exp

def canny_edge_detection(img):
    # Aplicar o detector de bordas de Canny
    edges = cv2.Canny(img, 300, 400)
    return edges

def dilate_image(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(img, kernel, iterations=1)
    return dilated


def pre_processing_image(img):
    img = intial_adjustments(img)
    img = apply_high_boost_filter(img, 1.5)
    # img = erode_image(img)
    # img = homomorphic_filter(img)
    # img = canny_edge_detection(img)
    img = dilate_image(img)
    # img = adjust_contrast_and_brightness(img, 1.5, 0)
    # img = reduce_noise(img)
    # img = equalize_histogram(img)
    return img