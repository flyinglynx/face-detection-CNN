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

def NMS(box):
    
    if len(box) == 0:
        return []
    
    #xmin, ymin, xmax, ymax, score, cropped_img, scale
    box.sort(key=lambda x :x[4])
    box.reverse()

    pick = []
    x_min = np.array([box[i][0] for i in range(len(box))],np.float32)
    y_min = np.array([box[i][1] for i in range(len(box))],np.float32)
    x_max = np.array([box[i][2] for i in range(len(box))],np.float32)
    y_max = np.array([box[i][3] for i in range(len(box))],np.float32)

    area = (x_max-x_min)*(y_max-y_min)
    idxs = np.array(range(len(box)))

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x_min[i],x_min[idxs[1:]])
        yy1 = np.maximum(y_min[i],y_min[idxs[1:]])
        xx2 = np.minimum(x_max[i],x_max[idxs[1:]])
        yy2 = np.minimum(y_max[i],y_max[idxs[1:]])

        w = np.maximum(xx2-xx1,0)
        h = np.maximum(yy2-yy1,0)

        overlap = (w*h)/(area[idxs[1:]] + area[i] - w*h)

        idxs = np.delete(idxs, np.concatenate(([0],np.where(((overlap >= 0.5) & (overlap <= 1)))[0]+1)))
    
    return [box[i] for i in pick]

def darwBoundingBox(boxList,img,color=(0,255,0)):
    
    for cords in boxList:
        x1,y1,x2,y2,s = cords
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0))
    
    return img