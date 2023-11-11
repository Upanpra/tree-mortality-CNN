# all functions used for CNN run
# calculating mean and standard deviation for normalizing the data
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# custom loader for more than 3 channel tif input
import tifffile


def my_tiff_loader(filename):
    return tifffile.imread(filename)


# Setting NAN to zero (particularly used for hyperspectral data)
def set_hyper_no_data_values_as_zero(data):
    data[data <  -3.39e38] = 0
    data[torch.isnan(data)] = 0
    data[torch.isinf(data)] = 0
    return data


def set_chm_no_data_values_as_zero(data):
    data[data < 0] = 0
    return data


def get_mean_std(train_folder, modality: str = "chm"):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.ImageFolder(train_folder, loader = my_tiff_loader, transform = transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for image_label in train_loader:
        data = image_label[0]
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        if modality == "hyper":
            data = set_hyper_no_data_values_as_zero(data)
        elif modality == "chm":
            data = data
        mean += torch.mean(data, 2).sum(0)
        std += torch.std(data, 2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


def load_model(net: torch.nn.Module, checkpoint: str) -> torch.nn.Module:
    if not torch.cuda.is_available():
        net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        net.load_state_dict(torch.load(checkpoint))
    return net

