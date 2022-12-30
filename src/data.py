import os
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def load_dataset(name, data_dir='./data', batch_size=16, num_workers=4):

    if name == 'cifar10':
        transforms = Compose(
            [ToTensor(),
             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = CIFAR10(root=data_dir,
                           train=True,
                           download=True,
                           transform=transforms)
        train_dataloader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers)
        testset = CIFAR10(root=data_dir,
                          train=False,
                          download=True,
                          transform=transforms)
        test_dataloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    else:
        train_dataloader, test_dataloader, classes = load_folder_dataset(data_dir, batch_size)

    return train_dataloader, test_dataloader, classes


def load_folder_dataset(data_dir, batch_size=16, num_workers=4):
    assert os.path.exists(f'{data_dir}/test')
    assert os.path.exists(f'{data_dir}/train')

    train_dataset = ImageFolderDataset(f'{data_dir}/train')
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataset = ImageFolderDataset(f'{data_dir}/test')
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return train_dataloader, test_dataloader, train_dataset.classes


class ImageFolderDataset(Dataset):
    '''
    Given a directory structure of:
        - Directory Name:
            - class_1:
                - sample_1.jpg
            - class_2:
                - sample_1.jpg
    Returns a dataset that iterates over each class
    '''

    def __init__(self, input_dir):
        super().__init__()

        self.f_paths = [
            f'{input_dir}/{cname}/{file}'
            for cname in os.listdir(input_dir)
            for file in os.listdir(f'{input_dir}/{cname}')
        ]

        self.transforms = Compose(
            [ToTensor(),
            Resize((256,256)),
             Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.classes = [cname for cname in os.listdir(input_dir)]
        self.classes = {cname: idx for idx, cname in enumerate(self.classes)}

    def __len__(self):
        return len(self.f_paths)

    def __getitem__(self, idx):
        path = self.f_paths[idx]
        img = Image.open(path)
        img = self.transforms(img)
        class_idx = self.classes[path.split("/")[-2]]
        return img, class_idx
        
