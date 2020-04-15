import cv2
import numpy as np
import os

import torch

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

'''
给出一张图片，判定是否为人脸
'''
class faceDetector():
    def __init__(self,model_path,width=100,height=100,model_name="forward CNN"):
        
        self.detectModel=None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.height=height
        self.width = width
        if model_name == "forward CNN":
            self.detectModel = torch.load(model_path)
            self.detectModel = self.detectModel.to(self.device)
    
    def detect(self,img):
        img = cv2.resize(img,(self.height,self.width),interpolation = cv2.INTER_AREA)
        img = img/255.0
        img = img.astype(np.float32)
        
        with torch.no_grad():
            inputTensor = torch.from_numpy(img)
            inputTensor = inputTensor.to(self.device)
            inputTensor= inputTensor.view(1,1,self.height,self.width)
            
            out = self.detectModel(inputTensor)
            out = out.cpu()
            out = out.numpy()
        
        return out[0]

