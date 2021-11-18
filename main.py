import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models
import torchvision
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def get_num_correct(preds, labels):
    # Количество правильных предиктов
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_data_loader(folder='train', batch_size=16):
    # Создаем датасет
    data_dir = os.path.join(os.getcwd(), folder)
    transform = T.Compose([
        T.Resize([47, 47]),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=(-20, 20)),
        T.ToTensor()
    ])

    data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True)

    return data_loader


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18(pretrained=True)

        self.model.fc = nn.Sequential(nn.Linear(
            self.model.fc.in_features, 3000), nn.ReLU(), nn.Linear(3000, num_classes))
        # Изменяем последний слой под нашу классификацию

    def forward(self, x):
        x = self.model(x)
        return x


def train_model(names):
    tb = SummaryWriter()
    net = Net(len(names))
    net.train()
    NEPOCH = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NEPOCH):
        x = get_data_loader(r'C:\nn\viola-jones\onlyfaces1')
        total_loss = 0
        total_correct = 0
        if epoch == int(NEPOCH/2):
            for param in list(net.model.parameters())[:-4]:
                param.requires_grad = False
        for imgs_batch, answers_batch in x:
            out = net(imgs_batch)
            optimizer.zero_grad()
            loss = criterion(out, answers_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += get_num_correct(out, answers_batch)

        tb.add_scalar("Loss", total_loss, epoch)
        tb.add_scalar("Correct", total_correct, epoch)
        tb.add_scalar("Accuracy", total_correct / len(x.dataset), epoch)
        tb.add_scalar("Check on test", raw_check(model=net), epoch)

    tb.close()
    return net


def main_for_train():
    nn = train_model(names)
    torch.save(nn.state_dict(), '6.model')


def raw_check(model=None):
    if model == None:
        model = Net(len(names))
        model.load_state_dict(torch.load('6.model'))
    model.eval()
    n = 9
    total_correct = 0
    for _ in range(n):
        tests = get_data_loader(r'C:\nn\viola-jones\onlyfacestest1', 1)
        for imgs_batch, answers_batch in tests:
            out = model(imgs_batch)
            check_predicts = get_num_correct(out, answers_batch)
            total_correct += check_predicts
    model.train()
    return total_correct/(len(tests.dataset)*n)


def check_for_one(path):
    transform = T.Compose([
        T.Resize([47, 47]),
        T.ToTensor()
    ])
    model = Net(len(names))
    model.load_state_dict(torch.load('6.model'))
    model.eval()

    original_image = cv2.imread(path)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    if detected_faces == ():
        return 'Нет лиц'
    (column, row, width, height) = detected_faces[0]
    d=30
    original_image = original_image[row-d:row +height+d, column-d:column+width+d]

            
    img =Image.fromarray(original_image[:,:,::-1])
    img = transform(img).unsqueeze(0)

    predict = model(img)

    answer = predict.argmax(dim=1)
    # print([(item,names[i]) for i,item in enumerate(predict.tolist()[0])])
    return names[answer.item()]


def get_names():
    data = get_data_loader('train')
    names = {}
    for path, index in data.dataset.samples:
        if index not in names:
            name = path.split('train\\')[-1].split('\\')[0]
            names[index] = name
    return names


if __name__ == '__main__':
    names = get_names()
    print(names)
    # main_for_train()
    print(f'Проверка на тестах:{raw_check()}')
    print('Фотографии из интернета:')
    for file in os.listdir('img_from_internet'):
        print(check_for_one("img_from_internet/"+file),file)
