import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import argparse
import numpy as np
from tqdm import tqdm
from dataloader_label import dataloader
from JointAE import JointAE
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from metrics_label import rmse, psnr, ssim, VIF
from math import exp
from PIL import Image


def rgb_to_ycrcb(rgb_tensor):
    rgb_to_ycrcb_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).T.cuda()

    shift = torch.tensor([0.0, 0.5, 0.5]).cuda()

    ycrcb = torch.tensordot(rgb_tensor.permute(0, 2, 3, 1), rgb_to_ycrcb_matrix, dims=1) + shift
    ycrcb = ycrcb.permute(0, 3, 1, 2)

    y = ycrcb[:, 0:1, :, :]
    crcb = ycrcb[:, 1:, :, :]

    return y, crcb


def ycrcb_to_rgb(y, crcb):
    ycrcb_to_rgb_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ]).T.cuda()

    shift = torch.tensor([0.0, -0.5, -0.5]).cuda()

    ycrcb = torch.cat([y, crcb], dim=1)
    ycrcb = ycrcb.permute(0, 2, 3, 1)

    rgb = torch.tensordot(ycrcb + shift, ycrcb_to_rgb_matrix, dims=1)
    rgb = rgb.permute(0, 3, 1, 2)

    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb


def rgb_y_cr_cb(rgb_tensor):
    rgb_to_ycrcb_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ]).T.cuda()

    shift = torch.tensor([0.0, 0.5, 0.5]).cuda()

    ycrcb = torch.tensordot(rgb_tensor.permute(0, 2, 3, 1), rgb_to_ycrcb_matrix, dims=1) + shift

    ycrcb = ycrcb.permute(0, 3, 1, 2)

    y = ycrcb[:, 0:1, :, :]
    cr = ycrcb[:, 1:2, :, :]
    cb = ycrcb[:, 2:3, :, :]

    return y, cr, cb


def y_cr_cb_rgb(y_fused, cr1, cr2, cb1, cb2):
    B, _, H, W = cb1.shape

    cb_fused = torch.ones((B, 1, H, W), device=cb1.device)
    cr_fused = torch.ones((B, 1, H, W), device=cb1.device)

    cb_diff1 = torch.abs(cb1 - 0.5)
    cb_diff2 = torch.abs(cb2 - 0.5)
    cr_diff1 = torch.abs(cr1 - 0.5)
    cr_diff2 = torch.abs(cr2 - 0.5)

    cb_mask = (cb_diff1 + cb_diff2) == 0
    cr_mask = (cr_diff1 + cr_diff2) == 0

    cb_fused[cb_mask] = 0.5
    cr_fused[cr_mask] = 0.5

    cb_fused[~cb_mask] = (
                                 cb1[~cb_mask] * cb_diff1[~cb_mask] + cb2[~cb_mask] * cb_diff2[~cb_mask]
                         ) / (cb_diff1[~cb_mask] + cb_diff2[~cb_mask])

    cr_fused[~cr_mask] = (
                                 cr1[~cr_mask] * cr_diff1[~cr_mask] + cr2[~cr_mask] * cr_diff2[~cr_mask]
                         ) / (cr_diff1[~cr_mask] + cr_diff2[~cr_mask])

    ycbcr_fused = torch.cat((y_fused, cr_fused, cb_fused), dim=1)

    ycbcr_to_rgb_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], device=y_fused.device).T.cuda()

    shift = torch.tensor([0.0, -0.5, -0.5], device=y_fused.device).view(1, 3, 1, 1).cuda()

    rgb = torch.tensordot((ycbcr_fused + shift), ycbcr_to_rgb_matrix, dims=([1], [0]))
    rgb = rgb.permute(0, 3, 1, 2)
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb


def save_and_visualize_image(output, save_dir, img_idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output = np.clip(output, 0, 1)
    img = (output * 255).astype(np.uint8)

    image = Image.fromarray(img)
    image_filename = os.path.join(save_dir, f'{img_idx:02d}.PNG')
    image.save(image_filename, format='PNG')

    print(f"Image saved as {image_filename}")


def Residual_Metric(output_metric, input_metric, style='sf'):
    if style == 'mse' or style == 'rmse':
        return exp(-(output_metric - input_metric)) - 1
    else:
        return exp(output_metric - input_metric) - 1


def model_test(task, model, dataloader_test, device):
    model.eval()
    test_rmse = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_vif = 0.0

    input_rmse = 0.0
    input_psnr = 0.0
    input_ssim = 0.0
    input_vif = 0.0
    with torch.no_grad():
        for batch_idx, (modal_A, modal_B, labels) in enumerate(dataloader_test):
            modal_A = modal_A.to(device)
            modal_B = modal_B.to(device)
            labels = labels.to(device)

            modal_A_y, cr_A, cb_A = rgb_y_cr_cb(modal_A)
            modal_B_y, cr_B, cb_B = rgb_y_cr_cb(modal_B)
            output_y = model(modal_A_y, modal_B_y)
            output = y_cr_cb_rgb(output_y, cr_A, cr_B, cb_A, cb_B)

            if task == 'MFI-Real':
                input_rmse += rmse(modal_A, labels)
                input_psnr += psnr(modal_A, labels)
                input_ssim += ssim(modal_A, labels)
                input_vif += VIF(modal_A, labels)
            elif task == 'MEF':
                input_rmse += rmse(modal_B, labels)
                input_psnr += psnr(modal_B, labels)
                input_ssim += ssim(modal_B, labels)
                input_vif += VIF(modal_B, labels)

            test_rmse += rmse(output, labels)
            test_psnr += psnr(output, labels)
            test_ssim += ssim(output, labels)
            test_vif += VIF(output, labels)

            input_rmse_all = input_rmse / len(dataloader_test)
            input_psnr_all = input_psnr / len(dataloader_test)
            input_ssim_all = input_ssim / len(dataloader_test)
            input_vif_all = input_vif / len(dataloader_test)

            test_rmse_all = test_rmse / len(dataloader_test)
            test_psnr_all = test_psnr / len(dataloader_test)
            test_ssim_all = test_ssim / len(dataloader_test)
            test_vif_all = test_vif / len(dataloader_test)

            res_rmse = Residual_Metric(test_rmse_all, input_rmse_all, style='rmse')
            res_psnr = Residual_Metric(test_psnr_all, input_psnr_all)
            res_ssim = Residual_Metric(test_ssim_all, input_ssim_all)
            res_vif = Residual_Metric(test_vif_all, input_vif_all)

        print(f'test_rmse:{test_rmse_all:.4f}, test_psnr:{test_psnr_all:.4f}, test_ssim:{test_ssim_all:.4f}, '
              f'test_vif:{test_vif_all:.4f}')
        print(f'input_rmse:{input_rmse_all:.4f}, input_psnr:{input_psnr_all:.4f}, input_ssim:{input_ssim_all:.4f}, '
              f'input_vif:{input_vif_all:.4f}')
        print(f'test_rmse:{test_rmse_all:.4f}, res_rmse:{res_rmse:.4f}, test_psnr:{test_psnr_all:.4f}, '
              f'res_psnr:{res_psnr:.4f}, test_ssim:{test_ssim_all:.4f}, res_ssim:{res_ssim:.4f}, '
              f'test_vif:{test_vif_all:.4f}, res_vif:{res_vif:.4f}')


def model_test_visualization():
    test_dataset_MFI_Real, test_dataset_MEF = dataloader()
    dataloader_test_MFI_Real = DataLoader(test_dataset_MFI_Real, batch_size=1, shuffle=False)
    dataloader_test_MEF = DataLoader(test_dataset_MEF, batch_size=1, shuffle=False)

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

    model.load_state_dict(torch.load(opt.model_path))
    print('The total number of parameters', count_parameters(model))

    print('MFI-Real:')
    model_test('MFI-Real', model, dataloader_test_MFI_Real, device)
    print('MEF:')
    model_test('MEF', model, dataloader_test_MEF, device)


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


# seed = 123
# seed_everything(seed=seed)

# torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--loss", type=str, default='MAE', help="MSE or MAE or CIL")
parser.add_argument("--input_size", type=int, default=1, help="")
parser.add_argument("--hidden_size", type=int, default=64, help="")
parser.add_argument("--AE_num_layers", type=int, default=4, help="")
parser.add_argument("--pool_size", type=int, default=64, help="")
parser.add_argument("--fs", type=float, default=0.2, help="")
parser.add_argument("--if_th", type=str, default='yes', help="yes or no")
parser.add_argument("--threshold", type=float, default=0.05, help="")
parser.add_argument("--gamma", type=float, default=0.7, help="")
parser.add_argument("--l_int", type=float, default=100.0, help="")
parser.add_argument("--l_ncc", type=float, default=1.0, help="")
parser.add_argument("--l_ssim", type=float, default=1.0, help="")
parser.add_argument("--l_grad", type=float, default=30.0, help="")
parser.add_argument("--optim", type=str, default='AdamW', help="Adam or AdamW or SGD")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of fmodal_Bst order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of fmodal_Bst order momentum of gradient")
parser.add_argument("--lr", type=float, default=0.0001, help="")
parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
parser.add_argument("--weight_decay", type=float, default=0.0005, help="")
parser.add_argument("--int_style", type=str, default='aa',
                    help="max or or mean or mask_n or mask_g or no_ir or aa")
parser.add_argument("--prior_sel", type=str, default='x_max', help="")
parser.add_argument("--model_sel", type=str, default='DRDB', help="RB or DB or RDB or ARDB or DRDB")
parser.add_argument("--model_path", type=str,
                    default='model/model.pth',
                    help="")
opt = parser.parse_args()
print(opt)

model_test_visualization()
