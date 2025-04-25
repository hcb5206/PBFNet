import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
from metrics import ssim


class ViIrImageDataset(Dataset):
    def __init__(self, vi_dir, ir_dir, transform=None):
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.transform = transform

        self.vi_images = sorted(os.listdir(vi_dir))
        self.ir_images = sorted(os.listdir(ir_dir))

        assert len(self.vi_images) == len(self.ir_images), \
            "vi和ir图像的数量不一致"

    def __len__(self):
        return len(self.vi_images)

    def __getitem__(self, idx):
        vi_img_path = os.path.join(self.vi_dir, self.vi_images[idx])
        ir_img_path = os.path.join(self.ir_dir, self.ir_images[idx])

        vi_img = Image.open(vi_img_path).convert('RGB')
        ir_img = Image.open(ir_img_path).convert('RGB')

        if self.transform:
            vi_img = self.transform(vi_img)
            ir_img = self.transform(ir_img)

        return vi_img, ir_img


def dataloader(test_dataset):
    transform = transforms.Compose([
        # transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        # transforms.Resize((high, width)),
        transforms.ToTensor(),
    ])

    dataset = ViIrImageDataset(
        vi_dir='train_dataset/vi',
        ir_dir='train_dataset/ir',
        transform=transform
    )
    if test_dataset == 'LLVIP':
        dataset_test = ViIrImageDataset(
            vi_dir='test_dataset/test/LLVIP/vi',
            ir_dir='test_dataset/test/LLVIP/ir',
            transform=transform_test
        )
    elif test_dataset == 'M3FD':
        dataset_test = ViIrImageDataset(
            vi_dir='test_dataset/test/M3FD/vi',
            ir_dir='test_dataset/test/M3FD/ir',
            transform=transform_test
        )
    elif test_dataset == 'MSRS':
        dataset_test = ViIrImageDataset(
            vi_dir='test_dataset/test/MSRS/vi',
            ir_dir='test_dataset/test/MSRS/ir',
            transform=transform_test
        )
    elif test_dataset == '0037':
        dataset_test = ViIrImageDataset(
            vi_dir='test_dataset/test_0037/vi',
            ir_dir='test_dataset/test_0037/ir',
            transform=transform_test
        )
    else:
        raise ValueError(f"test_dataset must be either 'LLVIP' or 'M3FD' or 'MSRS' or '0037'!")

    return dataset, dataset_test


# if __name__ == '__main__':
#     dataset, dataset_test = dataloader(test_dataset='M3FD')
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
#         axs[0].set_title('VI Image')
#         axs[0].axis('off')
#
#         axs[1].imshow(ir_img)
#         axs[1].set_title('IR Image')
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
#     for batch_idx, (vi, ir) in enumerate(dataloader):
#         print(vi.shape, ir.shape)
#         prior = torch.maximum(vi, ir)
#         # prior = 0.55 * vi + (1-0.55) * ir
#
#         print(f'ssim:{ssim(prior, vi, ir):.4f}')
#
#         show_images(vi, ir, prior, batch_idx)
