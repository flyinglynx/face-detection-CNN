import cv2
from lib.utils import faceDetector
import numpy as np

#faceDetector = faceDetector("trained_model/CNN64model.pt",width=64,height=64)
faceDetector = faceDetector("trained_model/CNN24model.pt",width=24,height=24)
cap = cv2.VideoCapture(0) #创建一个 VideoCapture 对象 

count = 0
while(cap.isOpened()):#循环读取每一帧
    bgr_image = cap.read()[1]
    bgr_image = cv2.flip(bgr_image,1,dst=None)
    
    h,w,c = bgr_image.shape
        
    ROI = bgr_image[int(h/2)-100:int(h/2)+100,int(w/2)-100:int(w/2)+100,:]
    cv2.imwrite("tempSample/pos_batch2_"+str(count)+".jpg",bgr_image[int(h/2)-100:int(h/2)+100,int(w/2)-100:int(w/2)+100,:])
    count=count+1
    #cv2.waitKey(0)
    #cv2.imwrite("new_pos_sample/pd_"+str(count)+".jpg",bgr_image[int(h/2)-100:int(h/2)+100,int(w/2)-100:int(w/2)+100,:])
    
    predict = faceDetector.detect(ROI)
    print(predict)
    if np.argmax(predict)==1:
        bgr_image = cv2.rectangle(bgr_image,(int(w/2)-100,int(h/2)-100),(int(w/2)+100,int(h/2)+100),(0,255,0))
    else:
        bgr_image = cv2.rectangle(bgr_image,(int(w/2)-100,int(h/2)-100),(int(w/2)+100,int(h/2)+100),(0,0,255))
    cv2.imshow('detection result', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() #释放摄像头
cv2.destroyAllWindows()#删除建立的全部窗口