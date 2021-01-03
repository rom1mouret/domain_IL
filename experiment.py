#!/usr/bin/env python3

import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from fruit_classification import (
    CNN, Classifier, FruitNet, Contract, MultiDistributionNet
)

if len(sys.argv) >= 2:
    device = sys.argv[1]  # e.g. cuda:0
else:
    device = 'cpu'

# problem specifications
n_classes1 = 118
n_classes2 = 13
latent_dim = 16
batch_size = 32

# pixel value normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# training function
def train(net: nn.Module, data_path: str, epochs: int):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    folder = ImageFolder(root=data_path, transform=transform)
    for epoch in range(1, epochs+1):
        dataloader = DataLoader(
            folder,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True
        )
        progress_bar = tqdm(dataloader, total=len(dataloader.dataset)//batch_size)
        for images, labels in progress_bar:
            if images.size(0) < 3:
                continue  # not good for batchnorm
            images = images.to(device)
            labels = labels.to(device)
            loss, report = net.loss(images, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            summary = epoch, loss.item(), report
            progress_bar.set_description("epoch: %i | loss: %.4f, %s" % summary)

# evaluating function
def accuracy(net: nn.Module, data_path: str) -> tuple:
    test_batch_size = 64  # no grad -> we can afford larger batch
    folder = ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(
        folder,
        batch_size=test_batch_size,
        num_workers=4,
        shuffle=False
    )
    hits = 0
    n = 0
    progress_bar = tqdm(dataloader, total=len(dataloader.dataset)//test_batch_size)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            pred = net(images)
            hits += (pred.max(dim=1)[1] == labels).float().sum().item()
        n += images.size(0)

    return hits, n, 100 * hits / n

# train the museum net and freeze it
museum_cnn = CNN(latent_dim).to(device)
museum_classifier = Classifier(latent_dim, n_classes=n_classes1).to(device)
museum_net = FruitNet(museum_classifier, museum_cnn)
print("training Museum Classifier and Museum CNN")
train(museum_net, "dataset/training/museum", epochs=3)
museum_cnn.freeze()
museum_classifier.freeze()

# train the contract
contract = Contract(museum_cnn, latent_dim).to(device)
print("training contract:")
train(contract, "dataset/training/museum", epochs=1)
contract.freeze()

# train a market net using museum's frozen classifier, but with a new CNN
market_cnn = CNN(latent_dim).to(device)
market_net = FruitNet(museum_classifier, market_cnn).comply_contract(contract)
print("training Market CNN")
train(market_net, "dataset/training/market", epochs=2)
market_cnn.freeze()

# test network merging
big_net = MultiDistributionNet(museum_classifier, contract, market_cnn, museum_cnn)
hits1, n1, acc1 = accuracy(big_net, "dataset/testing/museum")
hits2, n2, acc2 = accuracy(big_net, "dataset/testing/market")
print("merged network accuracy (museum):", acc1, "%")
print("merged network accuracy (market):", acc2, "%")
print("merged network total accuracy:", 100 * (hits1 + hits2) / (n1 + n2), "%")

# train a museum net on new classes, keeping the CNN trained for museum data
museum_0classifier = Classifier(latent_dim, n_classes=n_classes2).to(device)
museum_net = FruitNet(museum_0classifier, museum_cnn)
print("training Museum Classifier on new classes")
train(museum_net, "dataset/zeroshot/museum", epochs=3)
museum_0classifier.freeze()

# test the market net on the new classes using the trained museum classifier
market_net = FruitNet(museum_0classifier, market_cnn)
_, _, acc = accuracy(market_net, "dataset/zeroshot/market")
print("zero-shot accuracy on market data:", acc, "%")
