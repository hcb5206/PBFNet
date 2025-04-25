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
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from metrics_no_label import ssim, VIF, EN, SF, q_abf, AG
from math import exp

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_and_visualize_image(vi, save_dir, img_idx):
    vi = np.clip(vi, 0, 1)
    img = (vi * 255).astype(np.uint8)

    image = Image.fromarray(img)

    image_filename = os.path.join(save_dir, f'{img_idx:02d}.PNG')
    image.save(image_filename, format='PNG')
    print(f"Image saved as {image_filename}")

    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()


def Residual_Metric(vi_metric, input_metric, style='sf'):
    if style == 'mse' or style == 'rmse':
        return exp(-(vi_metric - input_metric)) - 1
    else:
        return exp(vi_metric - input_metric) - 1


# print(Residual_Metric(0.0075, 0.0198, 'rmse'))
# print(Residual_Metric(43.2378, 35.193))
# print(Residual_Metric(0.9921, 0.9498))
# print(Residual_Metric(0.9346, 0.8027))


def model_test(model, dataloader_test, device):
    model.eval()
    test_vif = 0.0
    test_en = 0.0
    test_ssim = 0.0
    test_sf = 0.0
    test_q_abf = 0.0
    test_ag = 0.0

    input_vif = 0.0
    input_en = 0.0
    input_ssim = 0.0
    input_sf = 0.0
    input_q_abf = 0.0
    input_ag = 0.0
    with torch.no_grad():
        for batch_idx, (near, far, labels) in enumerate(dataloader_test):
            near = near.to(device)
            far = far.to(device)
            labels = labels.to(device)

            output = model(near, far)
            # output = labels

            input_img = torch.maximum(near, far)

            input_vif += VIF(input_img, near, far)
            input_en += EN(input_img)
            input_ssim += ssim(input_img, near, far)
            input_sf += SF(input_img)
            input_q_abf += q_abf(input_img, near, far)
            input_ag += AG(input_img)

            test_vif += VIF(output, near, far)
            test_en += EN(output)
            test_ssim += ssim(output, near, far)
            test_sf += SF(output)
            test_q_abf += q_abf(output, near, far)
            test_ag += AG(output)

        input_vif_all = input_vif / len(dataloader_test)
        input_en_all = input_en / len(dataloader_test)
        input_ssim_all = input_ssim / len(dataloader_test)
        input_sf_all = input_sf / len(dataloader_test)
        input_q_abf_all = input_q_abf / len(dataloader_test)
        input_ag_all = input_ag / len(dataloader_test)

        test_vif_all = test_vif / len(dataloader_test)
        test_en_all = test_en / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)
        test_q_abf_all = test_q_abf / len(dataloader_test)
        test_ag_all = test_ag / len(dataloader_test)

        res_vif = Residual_Metric(test_vif_all, input_vif_all)
        res_en = Residual_Metric(test_en_all, input_en_all)
        res_ssim = Residual_Metric(test_ssim_all, input_ssim_all)
        res_sf = Residual_Metric(test_sf_all, input_sf_all)
        res_q_abf = Residual_Metric(test_q_abf_all, input_q_abf_all)
        res_ag = Residual_Metric(test_ag_all, input_ag_all)

    print(f'test_vif:{test_vif_all:.4f}, test_en:{test_en_all:.4f}, test_ssim:{test_ssim_all:.4f}, '
          f'test_sf:{test_sf_all:.4f}, test_q_abf:{test_q_abf_all:.4f}, test_ag:{test_ag_all:.4f}')
    print(f'input_vif:{input_vif_all:.4f}, input_en:{input_en_all:.4f}, input_ssim:{input_ssim_all:.4f}, '
          f'input_sf:{input_sf_all:.4f}, input_q_abf:{input_q_abf_all:.4f}, input_ag:{input_ag_all:.4f}')
    print(f'test_vif:{test_vif_all:.4f}, res_vif:{res_vif:.4f}, test_en:{test_en_all:.4f}, '
          f'res_en:{res_en:.4f}, test_ssim:{test_ssim_all:.4f}, res_ssim:{res_ssim:.4f}, '
          f'test_sf:{test_sf_all:.4f}, res_sf:{res_sf:.4f}, test_q_abf:{test_q_abf_all:.4f}, res_q_abf:{res_q_abf:.4f}, '
          f'test_ag:{test_ag_all:.4f}, res_ag:{res_ag:.4f}')


def model_test_visualization(dataset_test):
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    if opt.model_sel == 'RB':
        model = JointAE_res(input_size=opt.input_size, hidden_size=opt.hidden_size, AE_num_layers=opt.AE_num_layers,
                            pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
    elif opt.model_sel == 'DB':
        model = JointAE_dense(input_size=opt.input_size, hidden_size=opt.hidden_size, AE_num_layers=opt.AE_num_layers,
                              pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
    elif opt.model_sel == 'RDB':
        model = JointAE_res_dense(input_size=opt.input_size, hidden_size=opt.hidden_size,
                                  AE_num_layers=opt.AE_num_layers,
                                  pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
    elif opt.model_sel == 'ARDB':
        model = JointAE_agg_res_dense(input_size=opt.input_size, hidden_size=opt.hidden_size,
                                      AE_num_layers=opt.AE_num_layers,
                                      pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
    elif opt.model_sel == 'DRDB':
        model = JointAE(input_size=opt.input_size, hidden_size=opt.hidden_size,
                        AE_num_layers=opt.AE_num_layers,
                        pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
    else:
        raise ValueError(f"model_sel must be either 'RB' or 'DB' or 'RDB' or 'ARDB' or 'DRDB'!")
    print('The total number of parameters', count_parameters(model))
    model.load_state_dict(torch.load(opt.model_path))
    model_test(model=model, dataloader_test=dataloader_test, device=device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_test):
            near_focus = near_focus.to(device)
            far_focus = far_focus.to(device)
            labels = labels.to(device)

            output = model(near_focus, far_focus)
            # output = labels

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

            save_and_visualize_image(output_np, 'image', img_idx=batch_idx + 1)

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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--loss", type=str, default='MSE', help="MSE or MAE or CIL")
    parser.add_argument("--input_size", type=int, default=3, help="")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--AE_num_layers", type=int, default=4, help="")
    parser.add_argument("--pool_size", type=int, default=256, help="")
    parser.add_argument("--fs", type=float, default=0.2, help="")
    parser.add_argument("--threshold", type=float, default=0.05, help="")
    parser.add_argument("--gamma", type=float, default=0.7, help="")
    parser.add_argument("--optim", type=str, default='AdamW', help="Adam or AdamW or SGD")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lr", type=float, default=0.0005, help="")
    parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="")
    parser.add_argument("--patience", type=int, default=30, help="")
    parser.add_argument("--model_sel", type=str, default='DRDB', help="RB or DB or RDB or ARDB or DRDB")
    parser.add_argument("--prior_sel", type=str, default='xAB', help="")
    parser.add_argument("--model_path", type=str, default='model/model',
                        help="")
    opt = parser.parse_args()

    _, dataset_test = dataloader()

    model_test_visualization(dataset_test=dataset_test)
