import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors
import cv2

def resize_to_square(vegi_bgr):
    if vegi_bgr.shape[0] == 256 and vegi_bgr.shape[1] == 256:
        return vegi_bgr
    
    img_size = 256
    height, width = vegi_bgr.shape[:2]
    a1 = width / height
    a2 = height / width

    if (a1 > a2):
        r_img = cv2.resize(vegi_bgr, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
        margin = int(r_img.shape[1]/6)
        resized_img = r_img[0:img_size, margin:(margin+img_size)]

    elif(a1 < a2):
        # if height greater than width
        r_img = cv2.resize(vegi_bgr, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
        margin = int(r_img.shape[0]/6)
        resized_img = r_img[margin:(margin+img_size), 0:img_size]

    elif(a1 == a2):
        # if height and width are equal
        r_img = cv2.resize(vegi_bgr, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
        resized_img = r_img[0:img_size, 0:img_size]

    if(resized_img.shape[0] != img_size or resized_img.shape[1] != img_size):
        resized_img = r_img[0:img_size, 0:img_size]

    return resized_img


for i in range(1,5):
    img=cv2.imread("D:\Dokumente\Master-Studium\Semester3\KI_Projekt\Bilder Phillip\Phillip_Bilder\Zwiebel_Trieb/Zwiebel("+str(i)+").jpg")

    new_img=resize_to_square(img)


    cv2.imwrite("D:\Dokumente\Master-Studium\Semester3\KI_Projekt\Bilder Phillip\Phillip_Bilder\Zwiebel_Trieb/zwiebel_trieb_"+str(i)+".jpg",new_img)