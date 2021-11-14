from PIL import Image
import os
from numpy.lib.type_check import imag
from torch.functional import Tensor
import torch.multiprocessing as mp
import torch.optim as optim
import torch,random
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import numpy as np
from torchsummary import summary
NUM_CLASSES = 16
class MainNN(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

def trainmodel(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    count = 9000
    photos = os.listdir('yalefaces')
    epochs=100
    model.train()
    
    for _ in range(epochs):
        
        for sample in photos:
            answer  = int(sample.replace('subject','')[:2])
            image = Image.open(f'yalefaces\{sample}')
            image = image.resize((36,27))
            image = TF.to_grayscale(image)
            image = TF.to_tensor(image)
            image = image.unsqueeze(1)
            count += 1

            
           
           
            results =Tensor([answer]).to(torch.long)
            optimizer.zero_grad()
            outputs = model(image)
            outputslist= outputs.tolist()[0]
            loss = criterion(outputs, results)
            loss.backward()
            optimizer.step()
            loss= loss.item()
            
            print(count,loss)
    pat = f'{count}.model'
    torch.save(model.state_dict(), pat)


if __name__=='__main__':
    model= MainNN()
    model.load_state_dict(torch.load('9000.model'))
    trainmodel(model)
    