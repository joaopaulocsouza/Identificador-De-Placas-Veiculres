import cv2 
from matplotlib import pyplot as plt
import numpy as np
import imutils
import random
import os
import pytesseract

def recognition(image_path):
    print(image_path)
    img = cv2.imread(image_path) #read image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    plt.imshow(cv2.cvtColor(bfilter, cv2.COLOR_BGR2RGB)) #show processed image
    plt.title('Processed Image')

    edged = cv2.Canny(bfilter, 30, 200) #Edge detection
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    cv2.imshow('Text Detection edeg', edged) #show the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Find contours 
    contours = imutils.grab_contours(keypoints) #Grab contours 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #Sort contours

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
        
    print("Location: ", location)

    mask = np.zeros(gray.shape, np.uint8) #create blank image with same dimensions as the original image
    new_image = cv2.drawContours(mask, [location], 0,255, -1) #Draw contours on the mask image
    new_image = cv2.bitwise_and(img, img, mask=mask) #Take bitwise AND between the original image and mask image

    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)) #show the final image

    (x,y) = np.where(mask==255) #Find the co-ordinates of the four corners of the document
    (x1, y1) = (np.min(x), np.min(y)) #Find the top left corner
    (x2, y2) = (np.max(x), np.max(y)) #Find the bottom right corner
    cropped_image = gray[x1:x2+1, y1:y2+1] #Crop the image using the co-ordinates

    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)) #show the cropped image
    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
    text = pytesseract.image_to_string(cropped_image, lang='eng', config=config)
    font = cv2.FONT_HERSHEY_SIMPLEX #Font style
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA) #put the text on the image
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3) #Draw a rectangle around the text

    cv2.imshow('Text Detection', res) #show the image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)) #show the final image with text

def main():
    extensions = ['.png', '.jpg', '.jpeg']
    
    for file in os.listdir('./images'):
        path = os.path.join('./images', file)
        print(file)
        
        if os.path.isfile(path) and os.path.splitext(path)[1] in extensions:    
            recognition(path)
            
            # cv2.imshow('Imagem canny', detected)
            # cv2.waitKey(0)
        # cv2.destroyAllWindows()
main()