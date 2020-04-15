class SlidingWindow():
    def __init__(self,imgW,imgH,wW=200,wH=200,vStride=50,hStride=50):
        self.imgW = imgW       #图片宽度
        self.imgH = imgH       #图片高度
        self.wW = wW           #滑窗宽度
        self.wH = wH           #滑窗高度
        self.vStride = vStride #滑窗垂直方向上的步长
        self.hStride = hStride #滑窗水平方向上的步长
        self.last_pos = (0,0)  #上一次滑窗运行到的位置

    def resetWindow(self):
        self.last_pos = (0,0)
        
    def nextWindow(self,img):
        if self.last_pos == (self.imgH-self.wH,self.imgW-self.wW): #如果遍历完成
            return 0,0
        
        if self.last_pos[1] == self.imgW-self.wW:  #水平方向到顶
            if self.last_pos[0]+self.vStride <=  self.imgH-self.wH:   #且换行后不超出图像范围
                y = self.last_pos[0]+self.vStride                #换行扫描
                x = 0
            else:
                y = self.imgH-self.wH
                x = 0
            
            self.last_pos = (y,x)
            return img[y:y+self.wH,x:x+self.wW],[x,y,x+self.wW,y+self.wH]            #返回ROI区域    

        
        if self.last_pos[1]+self.hStride <= self.imgW-self.wW: 
            x = self.last_pos[1]+self.hStride
        else:             #水平方向即将到顶
            x = self.imgW-self.wW
        
        y = self.last_pos[0]
        self.last_pos = (y,x)
        return img[y:y+self.wH,x:x+self.wW],[x,y,x+self.wW,y+self.wH]            #返回ROI区域    