import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks.vgg import *
from networks.resnet import *
from datasets.data_loader import get_test_loader

test_loader = get_test_loader()

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


def test(epoch, model, device, criterion):
    print('\n[ Test epoch: %d ]' % epoch)
    model.eval()
    loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, targets).item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))

    print('\nTotal test accuracy:', 100. * correct / total)
    print('Total test loss:', loss)


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
    parser.add_argument('--file_name', type=str, help='name of the file to load the model')
    parser.add_argument('--loss_function', type=str, help='loss function to use')
    parser.add_argument('--epochs', type=int, help='number of epochs to train the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model_name
    file_name = args.file_name
    loss_function = args.loss_function
    epochs = args.epochs

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = available_models[model_name]().to(device)
    model.load_state_dict(torch.load(file_name))

    criterion = available_criteria[loss_function]()
    optimizer = available_optimizers['sgd'](model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch)
        test(epoch, model, device, criterion)

if __name__ == '__main__':
    main()
