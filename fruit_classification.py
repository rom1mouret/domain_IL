import torch
import torch.nn as nn

class FreezableNet(nn.Module):
    def freeze(self, state: bool=True) -> None:
        training = not state
        self.train(training)
        for p in self.parameters():
            p.requires_grad_(training)


class CNN(FreezableNet):
    def __init__(self, latent_dim: int) -> None:
        super(CNN, self).__init__()
        self._cnn = nn.Sequential(
            # block 1
            nn.Conv2d(3, 6*latent_dim, kernel_size=4),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(6*latent_dim),
            # block 2
            nn.Conv2d(6*latent_dim, 8*latent_dim, kernel_size=3, groups=latent_dim),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(8*latent_dim),
            # block 3
            nn.Conv2d(8*latent_dim, 16*latent_dim, kernel_size=3, groups=latent_dim),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(16*latent_dim),
            # block 3
            nn.Conv2d(16*latent_dim, 4 * latent_dim, kernel_size=3, groups=latent_dim),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(4*latent_dim),
            # block 4
            nn.Conv2d(4 *latent_dim, latent_dim, kernel_size=3, groups=latent_dim),
            # originally I put the ReLU here in order to facilitate merging networks
            # with operations like MAX, but this is no longer required.
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conved = self._cnn(x)
        # global average pooling
        pooled = conved.mean(dim=3).mean(dim=2)
        return pooled


class Contract(FreezableNet):
    """ currently: basic linear regression """
    def __init__(self, cnn: CNN, latent_dim: int) -> None:
        super(Contract, self).__init__()
        self._cnn = cnn
        self._contracts = nn.ModuleList([
            nn.Linear(latent_dim-1, 1) for i in range(latent_dim)
        ])
        self._indices = [
            list(range(i)) + list(range(i+1, latent_dim))
            for i in range(latent_dim)
        ]

    def compliance(self, x: torch.Tensor) -> torch.Tensor:
        pred = [
            contract(x[:, idx])
            for idx, contract in zip(self._indices, self._contracts)
        ]
        return -(torch.cat(pred, dim=1) - x).pow(2).mean(dim=1, keepdim=True)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        return -self.compliance(self._cnn(x)).mean(), ""



class Classifier(FreezableNet):
    def __init__(self, latent_dim: int, n_classes: int) -> None:
        super(Classifier, self).__init__()
        hidden_dim = (n_classes + latent_dim)//2
        self._decision = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, n_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decision(x)


class FruitNet(nn.Module):
    def __init__(self, classifier: Classifier, cnn: CNN) -> None:
        super(FruitNet, self).__init__()
        self._cnn = cnn
        self._classifier = classifier
        self._loss_function = nn.CrossEntropyLoss()
        self._contract = None

    def comply_contract(self, contract: Contract) -> "FruitNet":
        contract.freeze()
        self._contract = contract
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._classifier(self._cnn(x))

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        latent = self._cnn(x)
        pred = self._classifier(latent)
        loss = self._loss_function(pred, y)
        if self._contract is not None:
            loss = loss - self._contract.compliance(latent).mean()

        acc = 100 * (pred.max(dim=1)[1] == y).float().mean()
        report = "acc: %.1f" % acc

        return loss, report


class MultiDistributionNet(nn.Module):
    """ not suitable for training """
    def __init__(self, classifier: Classifier, contract: Contract, *cnn: CNN) -> None:
        super(MultiDistributionNet, self).__init__()
        self._cnns = list(cnn)
        self._classifier = classifier
        self._contract = contract

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = [cnn(x) for cnn in self._cnns]
        compliance = torch.cat(
            [self._contract.compliance(l) for l in latent], dim=1)
        # this trick only works when there are two CNNs:
        choice = compliance.max(dim=1, keepdim=True)[1].float()
        inp = latent[0] * (1 - choice) + latent[1] * choice

        return self._classifier(inp)
