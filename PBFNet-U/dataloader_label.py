import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
from metrics import ssim


class Unify_Dataset(Dataset):
    def __init__(self, A_dir, B_dir, label_dir, transform=None):
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.label_dir = label_dir
        self.transform = transform

        self.A_images = sorted(os.listdir(A_dir))
        self.B_images = sorted(os.listdir(B_dir))
        self.label_images = sorted(os.listdir(label_dir))

        assert len(self.A_images) == len(self.B_images) == len(self.label_images), \
            "modal_A和modal_B图像的数量不一致"

    def __len__(self):
        return len(self.A_images)

    def __getitem__(self, idx):
        A_img_path = os.path.join(self.A_dir, self.A_images[idx])
        B_img_path = os.path.join(self.B_dir, self.B_images[idx])
        label_img_path = os.path.join(self.label_dir, self.label_images[idx])

        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')
        label_img = Image.open(label_img_path).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
            label_img = self.transform(label_img)

        return A_img, B_img, label_img


def dataloader():
    transform_1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_2 = transforms.Compose([
        transforms.Resize((424, 624)),
        transforms.ToTensor(),
    ])
    test_dataset_MFI_Real = Unify_Dataset(
        A_dir='dataset/test/far',
        B_dir='dataset/test/near',
        label_dir='dataset/test/label',
        transform=transform_2
    )
    test_dataset_MEF = Unify_Dataset(
        A_dir='dataset/test/underexposed',
        B_dir='dataset/test/overexposed',
        label_dir='dataset/test/label',
        transform=transform_1
    )

    return test_dataset_MFI_Real, test_dataset_MEF
