import cv2
import numpy as np
import os
from pre_processing import pre_processing_image as pre_processing 
from detection import detection

def get_image_name(image_path):
    return image_path.split('/')[-1].split('.')[0]

def main():
    extensions = ['.png', '.jpg', '.jpeg']
    
    for file in os.listdir('./images'):
        path = os.path.join('./images', file)
        print(file)
        
        if os.path.isfile(path) and os.path.splitext(path)[1] in extensions:    
            img = cv2.imread(path)
            img_pre_processed = pre_processing(img)
            cv2.imwrite(f"./out/pre_process/pre_processed_{get_image_name(path)}.jpg", img_pre_processed)
            detection(img_pre_processed, img, get_image_name(path))
main()