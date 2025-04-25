import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
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
from LossFunction import M3Loss
from metrics import ssim, VIF, EN, SF, q_abf, AG
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
import matplotlib.pyplot as plt


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


def check_tensor_validity(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"{name} 包含 NaN 值！")
    else:
        print(f"{name} 不包含 NaN 值。")

    if torch.isinf(tensor).any():
        print(f"{name} 包含无穷值（Inf）！")
    else:
        print(f"{name} 不包含无穷值。")

    print(f"{name} 的最小值: {tensor.min().item()}, 最大值: {tensor.max().item()}")
    print(f"{name} 的形状: {tensor.shape}")


def model_test(task, model, dataloader_test, device):
    model.eval()
    test_vif = 0.0
    test_en = 0.0
    test_ssim = 0.0
    test_sf = 0.0
    test_q_abf = 0.0
    test_ag = 0.0
    with torch.no_grad():
        for batch_idx, (modal_A, modal_B) in enumerate(dataloader_test):
            modal_A = modal_A.to(device)
            modal_B = modal_B.to(device)

            if task == 'MFI-Real' or task == 'Lytro' or task == 'MEF':
                modal_A_y, cr_A, cb_A = rgb_y_cr_cb(modal_A)
                modal_B_y, cr_B, cb_B = rgb_y_cr_cb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = y_cr_cb_rgb(output_y, cr_A, cr_B, cb_A, cb_B)
            elif task == 'MSRS' or task == 'M3FD' or task == 'LLVIP' or task == 'MRIPET' or task == 'MRISPECT':
                modal_A_y, crcb_A = rgb_to_ycrcb(modal_A)
                modal_B_y, _ = rgb_to_ycrcb(modal_B)
                output_y = model(modal_A_y, modal_B_y)
                output = ycrcb_to_rgb(output_y, crcb_A)
            else:
                raise ValueError(f"Not in task selection range!")

            test_vif += VIF(output, modal_A, modal_B)
            test_en += EN(output)
            test_ssim += ssim(output, modal_A, modal_B)
            test_sf += SF(output)
            test_q_abf += q_abf(output, modal_A, modal_B)
            test_ag += AG(output)

        test_vif_all = test_vif / len(dataloader_test)
        test_en_all = test_en / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)
        test_q_abf_all = test_q_abf / len(dataloader_test)
        test_ag_all = test_ag / len(dataloader_test)
    return test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all


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


def train():
    seed = 123
    seed_everything(seed=seed)

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
    parser.add_argument("--model_path", type=str, default='model_train/model.pth',
                        help="")
    opt = parser.parse_args()
    print(opt)

    train_dataset, test_dataset_MFI_Real, test_dataset_Lytro, test_dataset_MEF, test_dataset_MSRS, \
    test_dataset_M3FD, test_dataset_LLVIP, test_dataset_MRIPET, test_dataset_MRISPECT, test_dataset_0037 = dataloader()
    dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
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
    # model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])

    # for name, param in model.named_parameters():
    #     print(name)

    print('The total number of parameters', count_parameters(model))

    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
        elif 'bias' in name:
            param.data.fill_(0.0)

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MFI-Real', model,
                                                                                                    dataloader_test_MFI_Real,
                                                                                                    device)
    print(f'MFI-Real:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')
    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('Lytro', model,
                                                                                                    dataloader_test_Lytro,
                                                                                                    device)
    print(f'Lytro:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')
    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MEF', model,
                                                                                                    dataloader_test_MEF,
                                                                                                    device)
    print(f'MEF:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MSRS', model,
                                                                                                    dataloader_test_MSRS,
                                                                                                    device)
    print(f'MSRS:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('M3FD', model,
                                                                                                    dataloader_test_M3FD,
                                                                                                    device)
    print(f'M3FD:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')
    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('LLVIP', model,
                                                                                                    dataloader_test_LLVIP,
                                                                                                    device)
    print(f'LLVIP:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')
    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MRIPET', model,
                                                                                                    dataloader_test_MRIPET,
                                                                                                    device)
    print(f'MRIPET:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')
    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MRISPECT', model,
                                                                                                    dataloader_test_MRISPECT,
                                                                                                    device)
    print(f'MRISPECT:'
          f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}, init test q_abf:{test_q_abf_all:.4f}, init test ag:{test_ag_all:.4f}')

    criterion = M3Loss(fs=opt.fs, loss=opt.loss, if_th=opt.if_th, threshold=opt.threshold, gamma=opt.gamma,
                       l_int=opt.l_int, l_ncc=opt.l_ncc, l_ssim=opt.l_ssim, l_grad=opt.l_grad,
                       int_style=opt.int_style).to(device)

    if opt.weight_decay_if == 'Yes':
        if opt.optim == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
                weight_decay=opt.weight_decay)
        elif opt.optim == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
                weight_decay=opt.weight_decay)
        elif opt.optim == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt.lr,
                momentum=0.9,
                weight_decay=opt.weight_decay)
        else:
            raise ValueError(f"optim must be either 'Adam' or 'AdamW' or 'SGD'!")
    elif opt.weight_decay_if == 'No':
        if opt.optim == 'Adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2))
        elif opt.optim == 'AdamW':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2))
        elif opt.optim == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt.lr,
                momentum=0.9)
        else:
            raise ValueError(f"optim must be either 'Adam' or 'AdamW' or 'SGD'!")

    else:
        raise ValueError(f"weight_decay_if must be either 'Yes' or 'No'!")

    for epoch in range(opt.n_epochs):
        model.train()
        running_loss = 0.0
        progress_bar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
        for batch_idx, (modal_A, modal_B) in enumerate(dataloader_train):
            modal_A = modal_A.to(device)
            modal_B = modal_B.to(device)

            modal_A_y, _ = rgb_to_ycrcb(modal_A)
            modal_B_y, _ = rgb_to_ycrcb(modal_B)

            output_y = model(modal_A_y, modal_B_y)

            loss = criterion(output_y, modal_A_y, modal_B_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar_train.update(1)
            progress_bar_train.set_postfix(loss=running_loss / len(dataloader_train))
        train_loss = running_loss / len(dataloader_train)
        progress_bar_train.close()
        print(f'Epoch {epoch + 1}/{opt.n_epochs}, train loss: {train_loss:.4f}')

        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MFI-Real',
                                                                                                        model,
                                                                                                        dataloader_test_MFI_Real,
                                                                                                        device)
        print(f'MFI-Real:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('Lytro', model,
                                                                                                        dataloader_test_Lytro,
                                                                                                        device)
        print(f'Lytro:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MEF', model,
                                                                                                        dataloader_test_MEF,
                                                                                                        device)
        print(f'MEF:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')

        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MSRS', model,
                                                                                                        dataloader_test_MSRS,
                                                                                                        device)
        print(f'MSRS:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')

        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('M3FD', model,
                                                                                                        dataloader_test_M3FD,
                                                                                                        device)
        print(f'M3FD:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('LLVIP', model,
                                                                                                        dataloader_test_LLVIP,
                                                                                                        device)
        print(f'LLVIP:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MRIPET', model,
                                                                                                        dataloader_test_MRIPET,
                                                                                                        device)
        print(f'MRIPET:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')
        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test('MRISPECT',
                                                                                                        model,
                                                                                                        dataloader_test_MRISPECT,
                                                                                                        device)
        print(f'MRISPECT:'
              f'test vif:{test_vif_all:.4f}, test en:{test_en_all:.4f}, test ssim:{test_ssim_all:.4f}, '
              f'test sf:{test_sf_all:.4f}, test q_abf:{test_q_abf_all:.4f}, test ag:{test_ag_all:.4f}')

        folder_path = os.path.join("model_train", f"{opt.model_sel}_{opt.prior_sel}_{opt.int_style}")
        os.makedirs(folder_path, exist_ok=True)
        file_name = f"epoch_{epoch + 1}.pth"
        file_path = os.path.join(folder_path, file_name)
        torch.save(model.state_dict(), file_path)

    #     if best_test_ssim <= test_ssim_all:
    #         best_test_ssim = test_ssim_all
    #         print(best_test_ssim)
    #         torch.save(model.state_dict(), opt.model_path)
    #         counter = 0
    #     else:
    #         counter += 1
    #
    #     if counter >= opt.patience:
    #         print("Early stopping")
    #         break
    #
    # print(best_test_ssim)
    # print("训练完成！")


if __name__ == "__main__":
    train()
