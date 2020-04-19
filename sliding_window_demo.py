import cv2
import numpy as np

import lib.SlidingWindow as sw
from lib.utils import faceDetector
import lib.utils as utils

#detector = faceDetector("trained_model/CNN64model.pt",width=64,height=64)
detector = faceDetector("trained_model/CNN24model.pt",width=24,height=24)
window = sw.SlidingWindow(imgW = 640,imgH = 480,wW = 150,wH = 150,vStride = 5,hStride=5)

cap = cv2.VideoCapture(0) #创建一个 VideoCapture 对象 

count = 0
while(cap.isOpened()):#循环读取每一帧
    bgr_image = cap.read()[1]
    bgr_image = cv2.flip(bgr_image,1,dst=None)
    
    boxes = detector.locateFace(bgr_image)
    utils.darwBoundingBox(boxes,bgr_image)

    cv2.imshow('detection result', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() #释放摄像头
cv2.destroyAllWindows()#删除建立的全部窗口
