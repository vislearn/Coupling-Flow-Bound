from math import prod

import torch
import torchvision
from sklearn.datasets import make_moons
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
import os


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class BasicDataset(Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def make_dataloader_moons(batch_size, train):
    moons = make_moons(6000, noise=.05,
                       random_state=235987 if train else 95832)
    data, target = map(torch.from_numpy, moons)
    dataset = BasicDataset(data.float(), target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def batch_mean_tr_cov(raw_byte_batch: torch.ByteTensor,
                      dequantization_noise: float):
    data = raw_byte_batch.to(torch.get_default_dtype()) / 255
    data = data.transpose(-1, -2)
    data = data + torch.randn_like(data) * dequantization_noise

    mean = torch.mean(data, dim=0, keepdim=True)
    cov = torch.cov(data.reshape(data.shape[0], -1).T, correction=0)
    cov_tr = torch.trace(cov)
    return mean, cov_tr


def make_dataloader_emnist(batch_size, normalized, dequantization_noise,
                           train=True, root_dir='./', pin_memory=True,
                           drop_last=False):
    transform = Compose([
        ToTensor(),
        Lambda(lambda x: x.view(1, 28, 28)),
        Lambda(lambda x: x.transpose(-1, -2)),
        AddGaussianNoise(std=dequantization_noise)
    ])
    try:
        emnist = torchvision.datasets.EMNIST(root=root_dir, split='digits',
                                             train=train, download=False,
                                             transform=transform)
    except RuntimeError:
        path = os.path.join(os.path.abspath(root_dir), 'EMNIST')
        yn = input(
            f'Dataset not found in {path}. '
            f'Would you like to download it here? (y/n): ')
        while True:
            if yn not in ['y', 'n']:
                yn = input('Please type \'y\' or \'n\': ')
            else:
                if yn == 'y':
                    emnist = torchvision.datasets.EMNIST(root=root_dir,
                                                         split='digits',
                                                         train=train,
                                                         download=True,
                                                         transform=transform)
                    break
                else:
                    print('Data will not be downloaded. Exiting script...')
                    raise ValueError("Data set not downloaded")

    if normalized:
        # Load data and apply equivalent transforms to above
        data = emnist.data
        dim = prod(data.shape[1:])
        mean, cov_tr = batch_mean_tr_cov(data, dequantization_noise)

        scale = (dim / cov_tr).sqrt()
        # Add this transform to the list
        transform.transforms.append(Lambda(lambda x: (x - mean) * scale))

    return torch.utils.data.DataLoader(
        emnist,
        batch_size=batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=1,
        drop_last=drop_last
    )


def make_dataloader(data_set, batch_size, normalized, dequantization,
                    train=True, root_dir='./', pin_memory=True, drop_last=False):
    if data_set == "EMNIST":
        return make_dataloader_emnist(
            batch_size, normalized, dequantization,
            train, root_dir, pin_memory, drop_last
        )
    elif data_set == "moons":
        return make_dataloader_moons(
            batch_size, normalized, dequantization,
            train
        )
    else:
        raise ValueError(f"Dataset {data_set!r} not known.")
