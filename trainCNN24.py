import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

import lib.CNN as cnn
import lib.utils as utils

'''
超参数
'''
BATCH_SIZE = 120
LR = 0.001
EPOCHS = 80
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
加载数据集(numpy数据类型)
'''
pos = utils.loadImages("positive",width=24,height=24,type_="RGB")
neg = utils.loadImages("negative",width=24,height=24,type_="RGB")
print("-------数据集装载完毕--------")

pos_y = np.array([1]*pos.shape[0]).reshape((-1,1))
neg_y = np.array([0]*neg.shape[0]).reshape((-1,1))

train_x = np.vstack((pos[100:,:,:,:],neg[500:986,:,:,:]))
val_x = np.vstack((pos[0:50,:,:,:],neg[0:500,:,:,:]))

train_y = np.vstack((pos_y[100:],neg_y[500:986]))
val_y = np.vstack((pos_y[0:50],neg_y[0:500]))

train_y = train_y.astype(np.int64)  #需要把标签转换为long型整数
val_y = val_y.astype(np.int64)
'''
将数据集包装为pytorch中的dataset类型
'''
train_x = torch.from_numpy(train_x)   #转换为torch的tensor
train_y = torch.from_numpy(train_y)

trainingSet = TensorDataset(train_x,train_y) #打包成Dataset类型
loader = DataLoader(dataset=trainingSet,batch_size=BATCH_SIZE,shuffle=True)

'''
开始训练
'''
model = cnn.CNN_24().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  

# Train the model
total_step = len(loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        labels= labels.view(labels.size()[0])
        # Forward pass

        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
        

'''
查看测试集的表现
'''
val_x = torch.from_numpy(val_x)   #转换为torch的tensor
val_y = torch.from_numpy(val_y)

validationSet = TensorDataset(val_x,val_y) #打包成Dataset类型
test_loader = torch.utils.data.DataLoader(dataset=validationSet,batch_size=BATCH_SIZE,shuffle=False)

with torch.no_grad():   #测试集中不需要计算梯度
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        labels= labels.view(labels.size()[0])
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy : {} %'.format(100 * correct / total))

if(input("save model? y/n:")=="y"):
    torch.save(model,'trained_model/CNN24model.pt')