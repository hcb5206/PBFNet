import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
from metrics import ssim


class MRI_PET_Dataset(Dataset):
    def __init__(self, mri_dir, pet_dir, transform=None):
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.transform = transform

        self.mri_images = sorted(os.listdir(mri_dir))
        self.pet_images = sorted(os.listdir(pet_dir))

        assert len(self.mri_images) == len(self.pet_images), \
            "mri和pet图像的数量不一致"

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        mri_img_path = os.path.join(self.mri_dir, self.mri_images[idx])
        pet_img_path = os.path.join(self.pet_dir, self.pet_images[idx])

        mri_img = Image.open(mri_img_path).convert('RGB')
        pet_img = Image.open(pet_img_path).convert('RGB')

        if self.transform:
            mri_img = self.transform(mri_img)
            pet_img = self.transform(pet_img)

        return mri_img, pet_img


class MRI_SPECT_Dataset(Dataset):
    def __init__(self, mri_dir, spect_dir, transform=None):
        self.mri_dir = mri_dir
        self.spect_dir = spect_dir
        self.transform = transform

        self.mri_images = sorted(os.listdir(mri_dir))
        self.spect_images = sorted(os.listdir(spect_dir))

        assert len(self.mri_images) == len(self.spect_images), \
            "mri和spect图像的数量不一致"

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        mri_img_path = os.path.join(self.mri_dir, self.mri_images[idx])
        spect_img_path = os.path.join(self.spect_dir, self.spect_images[idx])
        # print(mri_img_path, spect_img_path)

        mri_img = Image.open(mri_img_path).convert('RGB')
        spect_img = Image.open(spect_img_path).convert('RGB')

        if self.transform:
            mri_img = self.transform(mri_img)
            spect_img = self.transform(spect_img)

        return mri_img, spect_img


def dataloader_MRI_PET():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MRI_PET_Dataset(
        mri_dir='/home/he107552203884/MM_IMAGE_FUSION/Datasets/MED_datasets/MRI_PET/train/MRI',
        pet_dir='/home/he107552203884/MM_IMAGE_FUSION/Datasets/MED_datasets/MRI_PET/train/PET',
        transform=transform
    )
    dataset_test = MRI_PET_Dataset(
        mri_dir='test_dataset/PET/MRI',
        pet_dir='test_dataset/PET/PET',
        transform=transform_test
    )
    return dataset, dataset_test


def dataloader_MRI_SPECT():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MRI_SPECT_Dataset(
        mri_dir='/home/he107552203884/MM_IMAGE_FUSION/Datasets/MED_datasets/MRI_SPECT/train/MRI',
        spect_dir='/home/he107552203884/MM_IMAGE_FUSION/Datasets/MED_datasets/MRI_SPECT/train/SPECT',
        transform=transform
    )
    dataset_test = MRI_SPECT_Dataset(
        mri_dir='test_dataset/SPECT/MRI',
        spect_dir='test_dataset/SPECT/SPECT',
        transform=transform_test
    )
    return dataset, dataset_test


# if __name__ == '__main__':
#     dataset, dataset_test = dataloader_MRI_SPECT()
#
#     dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
#
#
#     def show_images(vi_img, ir_img, prior_img, batch_idx):
#         vi_img = vi_img.squeeze(0).permute(1, 2, 0).numpy()
#         ir_img = ir_img.squeeze(0).permute(1, 2, 0).numpy()
#         prior_img = prior_img.squeeze(0).permute(1, 2, 0).numpy()
#
#         fig, axs = plt.subplots(1, 3, figsize=(10, 5))
#
#         axs[0].imshow(vi_img)
#         axs[0].set_title('MRI Image')
#         axs[0].axis('off')
#
#         axs[1].imshow(ir_img)
#         axs[1].set_title('PET/SPECT Image')
#         axs[1].axis('off')
#
#         axs[2].imshow(prior_img)
#         axs[2].set_title('Prior Image')
#         axs[2].axis('off')
#
#         plt.suptitle(f'Batch {batch_idx}')
#         plt.tight_layout()
#         plt.show()
#
#
#     for batch_idx, (mri, pet) in enumerate(dataloader):
#         print(mri.shape, pet.shape)
#         # prior = torch.maximum(mri, pet)
#         # prior = 0.55 * mri + (1-0.55) * pet
#         prior = pet
#
#         print(f'ssim:{ssim(prior, mri, pet):.4f}')
#
#         show_images(mri, pet, prior, batch_idx)
