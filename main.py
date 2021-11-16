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
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


def get_num_correct(preds, labels):
    # Количество правильных предиктов
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_data_loader(folder='train', batch_size=6):
    # Создаем датасет
    data_dir = os.path.join(os.getcwd(), folder)
    transform = T.Compose([
        T.Resize([47, 47]),
        T.RandomHorizontalFlip(),
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
        # for param in self.model.parameters():
        #     # замораживаем слои, которые уже натренированы
        #     param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(
            self.model.fc.in_features, 3000), nn.ReLU(), nn.Linear(3000, num_classes))
        # Изменяем последний слой под нашу классификацию

    def forward(self, x):
        x = self.model(x)
        return x


def train_model(x, names):
    tb = SummaryWriter()
    net = Net(len(names))
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(20):
        total_loss = 0
        total_correct = 0
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

    tb.close()
    return net


def main():
    names = [
        'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Donald_Rumsfeld', 'Gerhard_Schroeder', 'Ariel_Sharon', 'Hugo_Chavez', 'Junichiro_Koizumi', 'Serena_Williams', 'John_Ashcroft', 'Jacques_Chirac', 'Vladimir_Putin'
    ]
    data = get_data_loader('train')
    nn = train_model(data, names)

    torch.save(nn.state_dict(), '3.model')


def check():
    names = [
        'George_W_Bush', 'Colin_Powell', 'Tony_Blair', 'Donald_Rumsfeld', 'Gerhard_Schroeder', 'Ariel_Sharon', 'Hugo_Chavez', 'Junichiro_Koizumi', 'Serena_Williams', 'John_Ashcroft', 'Jacques_Chirac', 'Vladimir_Putin'
    ]
    model = Net(len(names))
    model.load_state_dict(torch.load('3.model'))
    model.eval()

    tests = get_data_loader('train/', 1)
    total_correct = 0
    for imgs_batch, answers_batch in tests:
        out = model(imgs_batch)
        check_predicts = get_num_correct(out, answers_batch)

        if check_predicts != len(answers_batch):
            # Вывод неправильного предикта
            ind_incorrect = out.argmax(dim=1).eq(
                answers_batch).tolist().index(False)
            print(f"Оценка нейронки - {out[ind_incorrect].tolist()}")
            print(f'Правильный ответ: {answers_batch[ind_incorrect].item()}')

        total_correct += check_predicts

    print(total_correct/len(tests.dataset))


if __name__ == '__main__':
    # main()
    check()
