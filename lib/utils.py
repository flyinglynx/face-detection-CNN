import cv2
import numpy as np
import os

def loadImages(category,width,height):
    if category == "positive":
        path = "datasets/positive"
    elif category == "negative":
        path = "datasets/negative"
    else:
        print("invalid category")
        return None
    
    #读取文件夹下所有jpg图片的文件名
    names=[]
    for filename in os.listdir(path):
        if filename.endswith('jpg') :
            names.append(filename)
    
    images = []
    for filename in names:
        img = cv2.imread(path+'/'+filename,0)
        img = cv2.resize(img, (height,width),interpolation = cv2.INTER_AREA)
        img = img/255.0
        img = img.astype(np.float32)
        img=img.reshape(1,width,height)
        images.append(img)

    return np.array(images)
