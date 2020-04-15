import torch
import cv2
import numpy as np


#检查cuda是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detectModel = torch.load("trained_model/CNNmodel.pt")
model = detectModel.to(device)   #如果cuda可用，将模型改为GPU运算

testImg = cv2.imread("test.jpg",0)
testImg = cv2.resize(testImg,(100,100),interpolation = cv2.INTER_AREA)
testImg = testImg/255.0
testImg = testImg.astype(np.float32)

with torch.no_grad():
    inputTensor = torch.from_numpy(testImg)
    inputTensor = inputTensor.to(device)
    inputTensor= inputTensor.view(1,1,100,100)
    
    out = model(inputTensor)
    out = out.cpu()
    out = out.numpy()

print(out)
predict = np.argmax(out)