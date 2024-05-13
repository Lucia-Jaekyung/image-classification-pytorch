import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch import Tensor

from networks.vgg import *
from networks.resnet import *
from datasets.data_loader import get_train_loader


train_loader = get_train_loader()

available_models = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn
}

available_criteria = {
    'cross_entropy': nn.CrossEntropyLoss
}

available_optimizers = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}


def train(epoch, model, device, optimizer, criterion):
    print('\n[ Train epoch: %d ]' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current train average loss:', loss.item())

    print('\nTotal train accuracy:', 100. * correct / total)
    print('Total train loss:', train_loss)


def adjust_learning_rate(optimizer, epoch, learning_rate=0.1):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='name of the model to train')
    parser.add_argument('--file_name', type=str, help='name of the file to save the model')
    parser.add_argument('--loss_function', type=str, help='loss function to use')
    parser.add_argument('--optimizer', type=str, help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model_name
    file_name = args.file_name
    loss_function = args.loss_function
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    epochs = args.epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = available_models[model_name]().to(device)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    
    criterion = available_criteria[loss_function]()
    optimizer = available_optimizers[optimizer](model.parameters(), lr=learning_rate)

    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        train(epoch, model, device, optimizer, criterion)
        torch.save(model.state_dict(), file_name)

if __name__ == '__main__':
    main()