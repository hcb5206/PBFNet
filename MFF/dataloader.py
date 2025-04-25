import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MultiFocusImageDataset(Dataset):
    def __init__(self, near_focus_dir, far_focus_dir, labels_dir, transform=None):
        self.near_focus_dir = near_focus_dir
        self.far_focus_dir = far_focus_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.near_focus_images = sorted(os.listdir(near_focus_dir))
        self.far_focus_images = sorted(os.listdir(far_focus_dir))
        self.labels_images = sorted(os.listdir(labels_dir))

        assert len(self.near_focus_images) == len(self.far_focus_images) == len(self.labels_images), \
            "远焦、近焦和标签图像的数量不一致"

    def __len__(self):
        return len(self.near_focus_images)

    def __getitem__(self, idx):
        near_focus_img_path = os.path.join(self.near_focus_dir, self.near_focus_images[idx])
        far_focus_img_path = os.path.join(self.far_focus_dir, self.far_focus_images[idx])
        label_img_path = os.path.join(self.labels_dir, self.labels_images[idx])

        near_focus_img = Image.open(near_focus_img_path).convert('RGB')
        far_focus_img = Image.open(far_focus_img_path).convert('RGB')
        label_img = Image.open(label_img_path).convert('RGB')

        if self.transform:
            near_focus_img = self.transform(near_focus_img)
            far_focus_img = self.transform(far_focus_img)
            label_img = self.transform(label_img)
        return near_focus_img, far_focus_img, label_img


def dataloader():
    transform = transforms.Compose([
        transforms.Resize((424, 624)),
        transforms.ToTensor(),
    ])

    transform_lytro = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MultiFocusImageDataset(
        near_focus_dir='train_dataset/near',
        far_focus_dir='train_dataset/far',
        labels_dir='train_dataset/label',
        transform=transform
    )

    dataset_MFI_Real = MultiFocusImageDataset(
        near_focus_dir='test_dataset/MFI-Real/near',
        far_focus_dir='test_dataset/MFI-Real/far',
        labels_dir='test_dataset/MFI-Real/label',
        transform=transform
    )
    dataset_Lytro = MultiFocusImageDataset(
        near_focus_dir='test_dataset/Lytro/near',
        far_focus_dir='test_dataset/Lytro/far',
        labels_dir='test_dataset/Lytro/far',
        transform=transform_lytro
    )
    return dataset, dataset_MFI_Real, dataset_Lytro
