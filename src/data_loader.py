from functools import partial

import torch
import torchvision.transforms as transforms 
import tifffile
from torchvision import datasets
from typing import List, Sequence


def my_tiff_loader(filename):
    return tifffile.imread(filename)


# Setting NAN to zero (particularly used for hyperspectral data)
def set_hyper_no_data_values_as_zero(data):
    data[data < -3.39e38] = 0
    return data


def interpolate(x: torch.Tensor, input_size: Sequence[int]):
    # Perform torch.nn.functional.interpolate operation
    x = torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)
    return x


class ChannelSelectorTransform:

    def __init__(self, channel_indices: List[int]):
        print("using channel selector")

        self.channel_indices = channel_indices

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs should be of shape C X H X W
        assert isinstance(inputs, torch.Tensor), f"{type(inputs)} != torch.Tensor"
        assert len(inputs.shape) == 3, f"{inputs.shape} should be rank 3"
        res = inputs[self.channel_indices, :, :]
        return res


def transforms_aug(input_size, mean, std,channel_indices=None):
    interpolate_to_size = partial(interpolate, input_size=input_size)
    transform_train =[transforms.ToTensor(),
             transforms.Lambda(set_hyper_no_data_values_as_zero),
             transforms.Lambda(interpolate_to_size),
             transforms.Normalize(mean, std),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
             transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio = (0.8,1.2))
            ]

    if channel_indices is not None:
        transform_train.append(ChannelSelectorTransform(channel_indices))
    transform_train = transforms.Compose(transform_train)

    transform_test = [transforms.ToTensor(),
     transforms.Lambda(set_hyper_no_data_values_as_zero),
     transforms.Lambda(interpolate_to_size),
     transforms.Normalize(mean, std)
     ]

    if channel_indices is not None:
        transform_test.append(ChannelSelectorTransform(channel_indices))
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test
            
            
def data_loader(train_folder, test_folder, input_size, batch_size, mean, std, channel_indices=None):
    transform_train, transform_test = transforms_aug(input_size, mean, std, channel_indices=channel_indices)
    
    trainset = datasets.ImageFolder(train_folder, loader=my_tiff_loader, transform=transform_train)
    testset = datasets.ImageFolder(root = test_folder, transform = transform_test, loader=my_tiff_loader)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader
