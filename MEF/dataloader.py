import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class MultiFocusImageDataset(Dataset):
    def __init__(self, underexposed_dir, overexposed_dir, labels_dir, transform=None):
        self.underexposed_dir = underexposed_dir
        self.overexposed_dir = overexposed_dir
        self.labels_dir = labels_dir
        self.transform = transform

        self.underexposed_images = sorted(os.listdir(underexposed_dir))
        self.overexposed_images = sorted(os.listdir(overexposed_dir))
        self.labels_images = sorted(os.listdir(labels_dir))

        assert len(self.underexposed_images) == len(self.overexposed_images) == len(self.labels_images), \
            "低曝光、高曝光和标签图像的数量不一致"

    def __len__(self):
        return len(self.underexposed_images)

    def __getitem__(self, idx):
        underexposed_img_path = os.path.join(self.underexposed_dir, self.underexposed_images[idx])
        overexposed_img_path = os.path.join(self.overexposed_dir, self.overexposed_images[idx])
        label_img_path = os.path.join(self.labels_dir, self.labels_images[idx])

        underexposed_img = Image.open(underexposed_img_path).convert('RGB')
        overexposed_img = Image.open(overexposed_img_path).convert('RGB')
        label_img = Image.open(label_img_path).convert('RGB')

        if self.transform:
            underexposed_img = self.transform(underexposed_img)
            overexposed_img = self.transform(overexposed_img)
            label_img = self.transform(label_img)

        return underexposed_img, overexposed_img, label_img


def dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MultiFocusImageDataset(
        underexposed_dir='train_dataset/underexposed',
        overexposed_dir='train_dataset/overexposed',
        labels_dir='train_dataset/label',
        transform=transform
    )

    dataset_test = MultiFocusImageDataset(
        underexposed_dir='test_dataset/underexposed',
        overexposed_dir='test_dataset/overexposed',
        labels_dir='test_dataset/label',
        transform=transform
    )
    return dataset, dataset_test


# if __name__ == '__main__':
#     dataset, dataset_test = dataloader()
#
#     dataloader = DataLoader(dataset_test, batch_size=1, shuffle=False)
#
#
#     def show_images(underexposed_img, overexposed_img, label_img, batch_idx):
#         # output_dir = 'image'
#         underexposed_img = underexposed_img.squeeze(0).permute(1, 2, 0).numpy()
#         overexposed_img = overexposed_img.squeeze(0).permute(1, 2, 0).numpy()
#         label_img = label_img.squeeze(0).permute(1, 2, 0).numpy()
#
#         output = (underexposed_img + overexposed_img) / 2
#
#         fig, axs = plt.subplots(1, 4, figsize=(10, 5))
#
#         axs[0].imshow(underexposed_img)
#         axs[0].set_title('Underexposed Image')
#         axs[0].axis('off')
#
#         axs[1].imshow(overexposed_img)
#         axs[1].set_title('Overexposed Image')
#         axs[1].axis('off')
#
#         axs[2].imshow(label_img)
#         axs[2].set_title('Label Image')
#         axs[2].axis('off')
#
#         axs[3].imshow(output)
#         axs[3].set_title('Output Image')
#         axs[3].axis('off')
#
#         plt.suptitle(f'Batch {batch_idx}')
#         plt.tight_layout()
#         plt.show()
#
#         # save_path = os.path.join(output_dir, f'batch_{batch_idx}.png')
#         #
#         # plt.savefig(save_path, bbox_inches='tight')
#         #
#         # plt.close(fig)
#
#
#     for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader):
#         show_images(near_focus, far_focus, labels, batch_idx)
