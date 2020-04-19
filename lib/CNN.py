import torch.nn as nn
import torch.nn.functional as F


#级联CNN中最前面的一级，结构最简单
#输入的图像为三通道彩色图像,尺寸为24*24	
class CNN_24(nn.Module):
    def __init__(self):
        super(CNN_24, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.norm1 = nn.BatchNorm2d(64)		
        self.fc1 = nn.Linear(64 * 10* 10, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.norm1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#级联CNN中最前面的中间级
#输入的图像为三通道彩色图像,尺寸为64*64	
class CNN_64(nn.Module):
    def __init__(self):
        super(CNN_64, self).__init__()
        self.conv1 = nn.Conv2d(3,48,5)
        self.conv2 = nn.Conv2d(48,48,5)
        self.conv3 = nn.Conv2d(48,24,3)
        self.batchnorm = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(11*11*24, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 10)
        self.fc3 = nn.Linear(10, 2)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,2)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

