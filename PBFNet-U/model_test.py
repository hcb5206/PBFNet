import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import dataloader
from JointAE import JointAE
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from metrics import ssim, VIF, EN, SF, q_abf, AG
from math import exp
from PIL import Image
import csv
import time
from thop import profile


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
    image_filename = os.path.join(save_dir, f'{img_idx:02d}.png')
    image.save(image_filename, format='PNG')

    print(f"Image saved as {image_filename}")


def Residual_Metric(output_metric, input_metric, style='sf'):
    if style == 'mse' or style == 'rmse':
        return exp(-(output_metric - input_metric)) - 1
    else:
        return exp(output_metric - input_metric) - 1


# print(Residual_Metric(0.0065, 0.0198, 'rmse'))
# print(Residual_Metric(44.6042, 35.193))
# print(Residual_Metric(0.9929, 0.9498))
# print(Residual_Metric(0.9443, 0.8027))

# print(Residual_Metric(0.0712, 0.0544))
# print(Residual_Metric(0.2518, 0.2397))
# print(Residual_Metric(0.0252, 0.0185))


def model_test(task, model, dataloader_test, device):
    # output_csv_path = f"./results_csv/{task}.csv"
    #
    # if not os.path.exists(output_csv_path):
    #     with open(output_csv_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["Dataset", task])
    #         writer.writerow(["Sample Index", "VIF", "EN", "SSIM", "SF", "Q_ABF", "AG"])

    model.eval()
    test_vif = 0.0
    test_en = 0.0
    test_ssim = 0.0
    test_sf = 0.0
    test_q_abf = 0.0
    test_ag = 0.0

    # all_flops = 0.0
    # all_time = 0.0
    with torch.no_grad():
        for batch_idx, (modal_A, modal_B) in enumerate(dataloader_test):
            modal_A = modal_A.to(device)
            modal_B = modal_B.to(device)

            if task == 'MFI-Real' or task == 'Lytro' or task == 'MEF':
                # start_time = time.time()
                modal_A_y, cr_A, cb_A = rgb_y_cr_cb(modal_A)
                modal_B_y, cr_B, cb_B = rgb_y_cr_cb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = y_cr_cb_rgb(output_y, cr_A, cr_B, cb_A, cb_B)
                # end_time = time.time()
                # res_time = end_time - start_time
                # all_time += res_time
                #
                # flops, _ = profile(model, inputs=(modal_A_y, modal_B_y))
                # all_flops += flops / 1e9
            elif task == 'MSRS' or task == 'M3FD' or task == 'LLVIP' or task == '0037' or task == 'MRIPET' \
                    or task == 'MRISPECT':
                # start_time = time.time()
                modal_A_y, crcb_A = rgb_to_ycrcb(modal_A)
                modal_B_y, _ = rgb_to_ycrcb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = ycrcb_to_rgb(output_y, crcb_A)
                # end_time = time.time()
                # res_time = end_time - start_time
                # all_time += res_time
                #
                # flops, _ = profile(model, inputs=(modal_A_y, modal_B_y))
                # all_flops += flops / 1e9
            else:
                raise ValueError(f"Not in task selection range!")

            test_vif += VIF(output, modal_A, modal_B)
            test_en += EN(output)
            test_ssim += ssim(output, modal_A, modal_B)
            test_sf += SF(output)
            test_q_abf += q_abf(output, modal_A, modal_B)
            test_ag += AG(output)

            # with open(output_csv_path, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(
            #         [batch_idx + 1, test_vif / (batch_idx + 1), test_en / (batch_idx + 1), test_ssim / (batch_idx + 1),
            #          test_sf / (batch_idx + 1), test_q_abf / (batch_idx + 1), test_ag / (batch_idx + 1)])

        test_vif_all = test_vif / len(dataloader_test)
        test_en_all = test_en / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)
        test_q_abf_all = test_q_abf / len(dataloader_test)
        test_ag_all = test_ag / len(dataloader_test)

        # avg_flops = all_flops / len(dataloader_test)
        # avg_time = all_time / len(dataloader_test)
    return test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all


def visualization(task, model, dataloader_test, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (modal_A, modal_B) in enumerate(dataloader_test):
            modal_A = modal_A.to(device)
            modal_B = modal_B.to(device)

            if task == 'MFI-Real' or task == 'Lytro' or task == 'MEF':
                modal_A_y, cr_A, cb_A = rgb_y_cr_cb(modal_A)
                modal_B_y, cr_B, cb_B = rgb_y_cr_cb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = y_cr_cb_rgb(output_y, cr_A, cr_B, cb_A, cb_B)

                output_np = output.cpu().numpy().squeeze()
                output_np = output_np.transpose(1, 2, 0)

                save_and_visualize_image(output_np, f'image/{task}', img_idx=batch_idx + 1)
            elif task == 'MSRS' or task == 'M3FD' or task == 'LLVIP' or task == '0037' or task == 'MRIPET' \
                    or task == 'MRISPECT':
                modal_A_y, crcb_A = rgb_to_ycrcb(modal_A)
                modal_B_y, _ = rgb_to_ycrcb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = ycrcb_to_rgb(output_y, crcb_A)

                output_np = output.cpu().numpy().squeeze()
                output_np = output_np.transpose(1, 2, 0)

                save_and_visualize_image(output_np, f'image/{task}', img_idx=batch_idx + 1)
            else:
                raise ValueError(f"Not in task selection range!")


def model_test_visualization():
    _, test_dataset_MFI_Real, test_dataset_Lytro, test_dataset_MEF, test_dataset_MSRS, \
    test_dataset_M3FD, test_dataset_LLVIP, test_dataset_MRIPET, test_dataset_MRISPECT, test_dataset_0037 = dataloader()
    dataloader_test_MFI_Real = DataLoader(test_dataset_MFI_Real, batch_size=1, shuffle=False)
    dataloader_test_Lytro = DataLoader(test_dataset_Lytro, batch_size=1, shuffle=False)
    dataloader_test_MEF = DataLoader(test_dataset_MEF, batch_size=1, shuffle=False)
    dataloader_test_MSRS = DataLoader(test_dataset_MSRS, batch_size=1, shuffle=False)
    dataloader_test_M3FD = DataLoader(test_dataset_M3FD, batch_size=1, shuffle=False)
    dataloader_test_LLVIP = DataLoader(test_dataset_LLVIP, batch_size=1, shuffle=False)
    dataloader_test_MRIPET = DataLoader(test_dataset_MRIPET, batch_size=1, shuffle=False)
    dataloader_test_MRISPECT = DataLoader(test_dataset_MRISPECT, batch_size=1, shuffle=False)
    dataloader_test_0037 = DataLoader(test_dataset_0037, batch_size=1, shuffle=False)

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

    # visualization('0037', model, dataloader_test_0037, device)
    # visualization('MFI-Real', model, dataloader_test_MFI_Real, device)
    # visualization('Lytro', model, dataloader_test_Lytro, device)
    visualization('MEF', model, dataloader_test_MEF, device)
    # visualization('MSRS', model, dataloader_test_MSRS, device)
    # visualization('M3FD', model, dataloader_test_M3FD, device)
    # visualization('LLVIP', model, dataloader_test_LLVIP, device)
    # visualization('MRIPET', model, dataloader_test_MRIPET, device)
    # visualization('MRISPECT', model, dataloader_test_MRISPECT, device)

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('MFI-Real', model,
                   dataloader_test_MFI_Real,
                   device)
    print(f'MFI-Real:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('Lytro', model,
                   dataloader_test_Lytro,
                   device)
    print(f'Lytro:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('MEF', model,
                   dataloader_test_MEF,
                   device)
    print(f'MEF:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('MSRS', model,
                   dataloader_test_MSRS,
                   device)
    print(f'MSRS:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('M3FD', model,
                   dataloader_test_M3FD,
                   device)
    print(f'M3FD:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('LLVIP', model,
                   dataloader_test_LLVIP,
                   device)
    print(f'LLVIP:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('MRIPET', model,
                   dataloader_test_MRIPET,
                   device)
    print(f'MRIPET:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = \
        model_test('MRISPECT', model,
                   dataloader_test_MRISPECT,
                   device)
    print(f'MRISPECT:'
          f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
          f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')


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
