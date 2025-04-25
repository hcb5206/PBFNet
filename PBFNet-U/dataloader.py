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
    def __init__(self, A_dir, B_dir, transform=None):
        self.A_dir = A_dir
        self.B_dir = B_dir
        self.transform = transform

        self.A_images = sorted(os.listdir(A_dir))
        self.B_images = sorted(os.listdir(B_dir))

        assert len(self.A_images) == len(self.B_images), \
            "modal_A和modal_B图像的数量不一致"

    def __len__(self):
        return len(self.A_images)

    def __getitem__(self, idx):
        A_img_path = os.path.join(self.A_dir, self.A_images[idx])
        B_img_path = os.path.join(self.B_dir, self.B_images[idx])

        A_img = Image.open(A_img_path).convert('RGB')
        B_img = Image.open(B_img_path).convert('RGB')

        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return A_img, B_img


def dataloader():
    transform_1 = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_2 = transforms.Compose([
        transforms.Resize((424, 624)),
        transforms.ToTensor(),
    ])

    train_dataset = Unify_Dataset(
        A_dir='dataset/train/A',
        B_dir='dataset/train/B',
        transform=transform_1
    )
    test_dataset_MFI_Real = Unify_Dataset(
        A_dir='dataset/test/MFF/MFI-Real/far',
        B_dir='dataset/test/MFF/MFI-Real/near',
        transform=transform_2
    )
    test_dataset_Lytro = Unify_Dataset(
        A_dir='dataset/test/MFF/Lytro/far',
        B_dir='dataset/test/MFF/Lytro/near',
        transform=transform_1
    )
    test_dataset_MEF = Unify_Dataset(
        A_dir='dataset/test/underexposed',
        B_dir='dataset/test/overexposed',
        transform=transform_1
    )
    test_dataset_MSRS = Unify_Dataset(
        A_dir='dataset/test/VIF/MSRS/vi',
        B_dir='dataset/test/VIF/MSRS/ir',
        transform=transform_1
    )
    test_dataset_M3FD = Unify_Dataset(
        A_dir='dataset/test/VIF/M3FD/vi',
        B_dir='dataset/test/VIF/M3FD/ir',
        transform=transform_1
    )
    test_dataset_LLVIP = Unify_Dataset(
        A_dir='dataset/test/VIF/LLVIP/vi',
        B_dir='dataset/test/VIF/LLVIP/ir',
        transform=transform_1
    )
    test_dataset_MRIPET = Unify_Dataset(
        A_dir='dataset/test/MED/MRI-PET/PET-A',
        B_dir='dataset/test/MED/MRI-PET/MRI-B',
        transform=transform_1
    )
    test_dataset_MRISPECT = Unify_Dataset(
        A_dir='dataset/test/MED/MRI-SPECT/SPECT-A',
        B_dir='dataset/test/MED/MRI-SPECT/MRI-B',
        transform=transform_1
    )
    test_dataset_0037 = Unify_Dataset(
        A_dir='dataset/test_0037/vi',
        B_dir='dataset/test_0037/ir',
        transform=transform_1
    )

    return train_dataset, test_dataset_MFI_Real, test_dataset_Lytro, test_dataset_MEF, test_dataset_MSRS, \
           test_dataset_M3FD, test_dataset_LLVIP, test_dataset_MRIPET, test_dataset_MRISPECT, test_dataset_0037
