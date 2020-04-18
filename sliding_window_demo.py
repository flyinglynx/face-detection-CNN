import cv2
from lib.utils import faceDetector
import numpy as np
import lib.SlidingWindow as sw

detector = faceDetector("trained_model/CNNmodel.pt",width=100,height=100)
window = sw.SlidingWindow(imgW = 640,imgH = 480,wW = 200,wH = 200,vStride = 30,hStride=30)

cap = cv2.VideoCapture(0) #创建一个 VideoCapture 对象 

count = 0
while(cap.isOpened()):#循环读取每一帧
    bgr_image = cap.read()[1]
    bgr_image = cv2.flip(bgr_image,1,dst=None)
    
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)   #转换为灰度图做人脸检测
    h,w = gray_image.shape
    
    window.resetWindow()
    boundingBox = window.nextWindowPosition()
    
    while(boundingBox is not None):
        predict = detector.detect()        
        if np.argmax(predict)==1:
            bgr_image = cv2.rectangle(bgr_image,(cord[0],cord[1]),(cord[2],cord[3]),(0,255,0))
            #cv2.imshow("debug",ROI)
        ROI,cord = window.nextWindow(gray_image)
 
    cv2.imshow('detection result', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() #释放摄像头
cv2.destroyAllWindows()#删除建立的全部窗口
