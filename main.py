import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchsummary import summary


def create_dataset(names):
    x = []
    path = 'train/'
    for imgfolder in names:
        ind = names.index(imgfolder)
        for filename in os.listdir(path + imgfolder):
            filename = path + imgfolder + '/' + filename
            img = cv2.imread(filename, 0)
            img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_AREA)
            x.append((img, ind))
    return x


def prepare_photo(path):
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ToTensor()])

    
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_AREA)
    img = transform(torch.Tensor(img))
    imgtensor = img.unsqueeze(0)
    return imgtensor


def predict(path,net,names):
    img = prepare_photo(path)
    out= net(img)[0]
    out = [(names[i],float(item)) for i,item in enumerate(out)]
    namemax = max(out,key=lambda x: x[1])[0]
    return namemax

class Net(nn.Module):
    # def __init__(self, out):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.conv3 = nn.Conv2d(16, 32, 5)
    #     self.conv4 = nn.Conv2d(32, 64, 5)
    #     self.fc1 = nn.Linear(2048, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, out)

    # def forward(self, x):
    #     x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = F.max_pool2d(F.relu(self.conv4(x)), 2)
    #     x = torch.flatten(x, 1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def __init__(self,num_classes):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x


def train_model(x, names):

    net = Net(len(names))
    net.load_state_dict(torch.load('model1.md'))
    print(summary(net,(1,62,47)))
    net.train()
    objective = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0008)
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.ToTensor()])
    
    for ep in range(30):
        ls = []
        random.shuffle(x)
        for n, data in enumerate(x):

            imgtensor = transform(torch.Tensor(data[0]))
            imgtensor = torch.Tensor(imgtensor).unsqueeze(0)
            
            out = net(imgtensor)
            optimizer.zero_grad()
            answer = [-1]*len(names)
            answer[data[1]] = 1
            loss = objective(out[0], torch.Tensor(answer))
            loss.backward()
            optimizer.step()
            loss = loss.item()
            ls.append(loss)
        print(ep, sum(ls)/len(ls))
    return net


def main():
    names = [
        'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Donald_Rumsfeld', 'Gerhard_Schroeder', 'Ariel_Sharon', 'Hugo_Chavez', 'Junichiro_Koizumi', 'Serena_Williams', 'John_Ashcroft', 'Jacques_Chirac', 'Vladimir_Putin'
    ]
    data = create_dataset(names)
    nn = train_model(data, names)
    
    torch.save(nn.state_dict(), 'model.md')


def check():
    names = [
        'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Donald_Rumsfeld', 'Gerhard_Schroeder', 'Ariel_Sharon', 'Hugo_Chavez', 'Junichiro_Koizumi', 'Serena_Williams', 'John_Ashcroft', 'Jacques_Chirac', 'Vladimir_Putin'
    ]
    net = Net(len(names))
    net.load_state_dict(torch.load('model.md'))
    net.eval()
    print(predict('Bush.jpg',net,names))
    print(predict('vladimir.jpeg',net,names))

    
    


if __name__ == '__main__':
    # main()
    check()
