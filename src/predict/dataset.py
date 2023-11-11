import os
import torch
import torchvision.transforms as transforms
from typing import Optional, Any, Dict, List, Tuple
from torch.utils.data import Dataset
from src.functions import my_tiff_loader, set_hyper_no_data_values_as_zero


class PredictFolderDataset(Dataset):
    """Load all images in a folder."""

    def __init__(self, input_folder: str, transform=None, ext: Optional[Tuple[str]] = (".tif", ".TIF")):
        """
        Args:
            input_folder: str: path to the input folder of images to load
            transform (callable, optional): Optional transform to be applied
                on a sample.
            ext: if non-None, only load files matching this extension
        """
        self.input_folder = input_folder
        self.images = os.listdir(self.input_folder)
        if ext is not None:
            self.images = [x for x in self.images if any(x.endswith(ex) for ex in ext)]
        self.transform = transform
        print(f"Built PredictFolderDataset of length {len(self)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.input_folder, self.images[idx])
        image = my_tiff_loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        sample = {'image': image, 'filename': self.images[idx], "full_input_path": img_name}

        return sample


def transforms_aug(input_size, mean, std, channel_indices=None):
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
         transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
         transforms.Normalize(mean, std),
         # transforms.ToPILImage(),
         transforms.RandomVerticalFlip(p=0.5),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.GaussianBlur(kernel_size = 3, sigma=(1e-7, 1.0)),
         transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio = (0.8,1.2))
         ])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(lambda x: set_hyper_no_data_values_as_zero(x)),
         transforms.Lambda(lambda x: torch.nn.functional.interpolate(x.unsqueeze(0), tuple(input_size)).squeeze(0)),
         transforms.Normalize(mean, std)
         ])

    return transform_train, transform_test


def get_predict_loader(tiff_input_folder: str, input_size, mean, std, batch_size: int):
    transform_train, transform_test = transforms_aug(input_size, mean, std)

    dataset = PredictFolderDataset(tiff_input_folder, transform_test)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader
