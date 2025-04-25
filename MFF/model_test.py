import os
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
from dataloader import dataloader
from JointAE import JointAE
from LossFunction import M3Loss
from metrics import rmse, mse, psnr, ssim, VIF, ms_ssim, SF, SD, AG
from math import exp


def save_and_visualize_image(output, save_dir, img_idx):
    output = np.clip(output, 0, 1)
    img = (output * 255).astype(np.uint8)

    image = Image.fromarray(img)

    image_filename = os.path.join(save_dir, f'{img_idx:02d}.PNG')
    image.save(image_filename, format='PNG')
    print(f"Image saved as {image_filename}")

    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()


def Residual_Metric(output_metric, input_metric, style='sf'):
    if style == 'mse' or style == 'rmse':
        return exp(-(output_metric - input_metric)) - 1
    else:
        return exp(output_metric - input_metric) - 1


def model_test(model, dataloader_test, device):
    model.eval()
    test_rmse = 0.0
    test_mse = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_vif = 0.0
    test_ms_ssim = 0.0
    test_sf = 0.0
    test_sd = 0.0
    test_ag = 0.0

    input_rmse = 0.0
    input_psnr = 0.0
    input_ssim = 0.0
    input_vif = 0.0
    input_sf = 0.0
    input_sd = 0.0
    input_ag = 0.0
    with torch.no_grad():
        for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_test):
            near_focus = near_focus.to(device)
            far_focus = far_focus.to(device)
            labels = labels.to(device)

            output = model(near_focus, far_focus)

            input_rmse += rmse(far_focus, labels)
            input_psnr += psnr(far_focus, labels)
            input_ssim += ssim(far_focus, labels)
            input_vif += VIF(far_focus, labels)
            input_sf += SF(far_focus)
            input_sd += SD(far_focus)
            input_ag += AG(far_focus)

            test_rmse += rmse(output, labels)
            test_mse += mse(output, labels)
            test_psnr += psnr(output, labels)
            test_ssim += ssim(output, labels)
            test_vif += VIF(output, labels)
            test_ms_ssim += ms_ssim(output, labels)
            test_sf += SF(output)
            test_sd += SD(output)
            test_ag += AG(output)

        input_rmse_all = input_rmse / len(dataloader_test)
        input_psnr_all = input_psnr / len(dataloader_test)
        input_ssim_all = input_ssim / len(dataloader_test)
        input_vif_all = input_vif / len(dataloader_test)
        input_sf_all = input_sf / len(dataloader_test)
        input_sd_all = input_sd / len(dataloader_test)
        input_ag_all = input_ag / len(dataloader_test)

        test_rmse_all = test_rmse / len(dataloader_test)
        test_mse_all = test_mse / len(dataloader_test)
        test_psnr_all = test_psnr / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_vif_all = test_vif / len(dataloader_test)
        test_ms_ssim_all = test_ms_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)
        test_sd_all = test_sd / len(dataloader_test)
        test_ag_all = test_ag / len(dataloader_test)

        res_rmse = Residual_Metric(test_rmse_all, input_rmse_all, style='rmse')
        res_psnr = Residual_Metric(test_psnr_all, input_psnr_all)
        res_ssim = Residual_Metric(test_ssim_all, input_ssim_all)
        res_vif = Residual_Metric(test_vif_all, input_vif_all)
        res_sf = Residual_Metric(test_sf_all, input_sf_all)
        res_sd = Residual_Metric(test_sd_all, input_sd_all)
        res_ag = Residual_Metric(test_ag_all, input_ag_all)

    print(f'test_rmse:{test_rmse_all:.4f}, test_mse:{test_mse_all:.4f}, test_psnr:{test_psnr_all:.4f}, '
          f'test_ssim:{test_ssim_all:.4f}, test_vif:{test_vif_all:.4f}, test_ms_ssim:{test_ms_ssim_all:.4f}, '
          f'test_sf:{test_sf_all:.4f}, test_sd:{test_sd_all:.4f}, test_ag:{test_ag_all:.4f}')
    print(f'input_rmse:{input_rmse_all:.4f}, input_psnr:{input_psnr_all:.4f}, input_ssim:{input_ssim_all:.4f}, '
          f'input_vif:{input_vif_all:.4f}, input_sf:{input_sf_all:.4f}, input_sd:{input_sd_all:.4f}, '
          f'input_ag:{input_ag_all:.4f}')
    print(f'test_rmse:{test_rmse_all:.4f}, res_rmse:{res_rmse:.4f}, test_psnr:{test_psnr_all:.4f}, '
          f'res_psnr:{res_psnr:.4f}, test_ssim:{test_ssim_all:.4f}, res_ssim:{res_ssim:.4f}, '
          f'test_vif:{test_vif_all:.4f}, res_vif:{res_vif:.4f}, test_sf:{test_sf_all:.4f}, res_sf:{res_sf:.4f}, '
          f'test_sd:{test_sd_all:.4f}, res_sd:{res_sd:.4f}, test_ag:{test_ag_all:.4f}, res_ag:{res_ag:.4f}')


def model_test_visualization(model_path, dataset_test, modal_sel):
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    model = JointAE(input_size=opt.input_size, hidden_size=opt.hidden_size, AE_num_layers=opt.AE_num_layers,
                    pool_size=opt.pool_size, modal_sel=modal_sel).to(device)
    print('The total number of parameters', count_parameters(model))
    model.load_state_dict(torch.load(model_path))
    model_test(model=model, dataloader_test=dataloader_test, device=device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_test):
            near_focus = near_focus.to(device)
            far_focus = far_focus.to(device)
            labels = labels.to(device)

            output = model(near_focus, far_focus)

            # x_near_spatial_r = normalize_image(x_near_spatial_r)
            # x_far_spatial_r = normalize_image(x_far_spatial_r)
            # spatial_com = normalize_image(spatial_com)
            # output = normalize_image(output)

            far_focus_np = far_focus.cpu().numpy().squeeze()
            near_focus_np = near_focus.cpu().numpy().squeeze()
            labels_np = labels.cpu().numpy().squeeze()
            output_np = output.cpu().numpy().squeeze()

            far_focus_np = far_focus_np.transpose(1, 2, 0)
            near_focus_np = near_focus_np.transpose(1, 2, 0)
            labels_np = labels_np.transpose(1, 2, 0)
            output_np = output_np.transpose(1, 2, 0)

            # save_and_visualize_image(output_np, 'image', img_idx=batch_idx + 1)

            # save_dir = 'C:\\Users\\HE CONG BING\\Desktop\\contrastive algorithm code\\SFINet\\result\\res\\Lytro'
            # save_output_image(output_np, save_dir, batch_idx + 1)

            # fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            #
            # axes[0, 0].imshow(near_focus_np)
            # axes[0, 0].set_title('Near Focus')
            # axes[0, 0].axis('off')
            #
            # axes[0, 1].imshow(far_focus_np)
            # axes[0, 1].set_title('Far Focus')
            # axes[0, 1].axis('off')
            #
            # axes[1, 0].imshow(labels_np)
            # axes[1, 0].set_title('Labels')
            # axes[1, 0].axis('off')
            #
            # axes[1, 1].imshow(output_np)
            # axes[1, 1].set_title('Output')
            # axes[1, 1].axis('off')
            #
            # plt.tight_layout()
            # plt.show()


def normalize_image(image: torch.Tensor, new_range=(0, 1)) -> torch.Tensor:
    min_val, max_val = new_range
    image = image.float()
    img_min = image.min()
    img_max = image.max()
    normalized_image = (image - img_min) / (img_max - img_min)
    normalized_image = normalized_image * (max_val - min_val) + min_val

    return normalized_image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# seed = 360
# seed_everything(seed=seed)

# torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="")
parser.add_argument("--batch_size", type=int, default=4, help="")
parser.add_argument("--loss", type=str, default='MSE', help="MSE or MAE or CIL")
parser.add_argument("--input_size", type=int, default=3, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
parser.add_argument("--high", type=int, default=424, help="")
parser.add_argument("--width", type=int, default=624, help="")
parser.add_argument("--AE_num_layers", type=int, default=4, help="")
parser.add_argument("--pool_size", type=int, default=64, help="")
parser.add_argument("--fs", type=float, default=0.2, help="")
parser.add_argument("--threshold", type=float, default=0.2, help="")
parser.add_argument("--gamma", type=float, default=0.45, help="")
parser.add_argument("--optim", type=str, default='AdamW', help="Adam or AdamW or SGD")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lr", type=float, default=0.0005, help="")
parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
parser.add_argument("--weight_decay", type=float, default=0.0005, help="")
parser.add_argument("--patience", type=int, default=30, help="")
parser.add_argument("--modal_sel", type=str, default='xB', help="")
parser.add_argument("--model_path", type=str, default='model/model',
                    help="")
opt = parser.parse_args()
print(opt)

dataset, dataset_test, _ = dataloader()

model_test_visualization(model_path=opt.model_path, dataset_test=dataset_test, modal_sel=opt.modal_sel)
