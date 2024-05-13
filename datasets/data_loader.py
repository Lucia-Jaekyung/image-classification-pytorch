import torch
import torchvision
import torchvision.transforms as transforms

def get_train_loader(batch_size=128):
    """
    CIFAR-10 data loader
    input: batch_size
    output: train_loader
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader


def get_test_loader():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    return test_loader