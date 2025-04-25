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
from dataloader import dataloader_MRI_PET, dataloader_MRI_SPECT
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


def model_test(model, dataloader_test, device):
    model.eval()
    test_vif = 0.0
    test_en = 0.0
    test_ssim = 0.0
    test_sf = 0.0
    test_q_abf = 0.0
    test_ag = 0.0
    with torch.no_grad():
        for batch_idx, (mri, pet) in enumerate(dataloader_test):
            mri = mri.to(device)
            pet = pet.to(device)

            # mri_y, crcb_mri = rgb_to_ycrcb(mri)
            # pet_y, crcb_pet = rgb_to_ycrcb(pet)

            output = model(mri, pet)

            # output = ycrcb_to_rgb(output, crcb_pet)

            test_vif += VIF(output, mri, pet)
            test_en += EN(output)
            test_ssim += ssim(output, mri, pet)
            test_sf += SF(output)
            test_q_abf += q_abf(output, mri, pet)
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
    parser.add_argument("--input_size", type=int, default=3, help="")
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
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of fpetst order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of fpetst order momentum of gradient")
    parser.add_argument("--lr", type=float, default=0.0005, help="")
    parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="")
    parser.add_argument("--patience", type=int, default=30, help="")
    parser.add_argument("--int_style", type=str, default='max',
                        help="max or or mean or mask_n or mask_g or no_ir or aa")
    parser.add_argument("--prior_sel", type=str, default='x_max', help="")
    parser.add_argument("--model_sel", type=str, default='DRDB', help="RB or DB or RDB or ARDB or DRDB")
    parser.add_argument("--tasks", type=str, default='PET', help="PET or SPECT")
    parser.add_argument("--model_path", type=str, default='model_train/model.pth',
                        help="")
    opt = parser.parse_args()

    if opt.tasks == 'PET':
        dataset, dataset_test = dataloader_MRI_PET()
    else:
        dataset, dataset_test = dataloader_MRI_SPECT()
    dataloader_train = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    print(opt)

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

    # model.load_state_dict(torch.load(opt.model_path))

    # for name, param in model.named_parameters():
    #     print(name)

    print('The total number of parameters', count_parameters(model))

    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
        elif 'bias' in name:
            param.data.fill_(0.0)

    test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test(model,
                                                                                                    dataloader_test,
                                                                                                    device)
    print(f'init test vif:{test_vif_all:.4f}, init test en:{test_en_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
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

    # best_test_ssim = -np.inf
    # counter = 0
    for epoch in range(opt.n_epochs):
        model.train()
        running_loss = 0.0
        train_vif = 0.0
        train_en = 0.0
        train_ssim = 0.0
        train_sf = 0.0
        train_q_abf = 0.0
        train_ag = 0.0
        progress_bar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
        for batch_idx, (mri, pet) in enumerate(dataloader_train):
            mri = mri.to(device)
            pet = pet.to(device)

            # mri_y, crcb_mri = rgb_to_ycrcb(mri)
            # pet_y, crcb_pet = rgb_to_ycrcb(pet)

            output = model(mri, pet)

            loss = criterion(output, mri, pet)
            optimizer.zero_grad()
            loss.backward()
            # check_tensor_validity(model.encoder_A.conv_res.weight.grad, 'weigh_grad')
            optimizer.step()

            # output = ycrcb_to_rgb(output_y, crcb_pet)

            train_vif += VIF(output, mri, pet)
            train_en += EN(output)
            train_ssim += ssim(output, mri, pet)
            train_sf += SF(output)
            train_q_abf += q_abf(output, mri, pet)
            train_ag += AG(output)

            running_loss += loss.item()
            progress_bar_train.update(1)
            progress_bar_train.set_postfix(loss=running_loss / len(dataloader_train))
        train_loss = running_loss / len(dataloader_train)
        train_vif_all = train_vif / len(dataloader_train)
        train_en_all = train_en / len(dataloader_train)
        train_ssim_all = train_ssim / len(dataloader_train)
        train_sf_all = train_sf / len(dataloader_train)
        train_q_abf_all = train_q_abf / len(dataloader_train)
        train_ag_all = train_ag / len(dataloader_train)
        progress_bar_train.close()

        test_vif_all, test_en_all, test_ssim_all, test_sf_all, test_q_abf_all, test_ag_all = model_test(model,
                                                                                                        dataloader_test,
                                                                                                        device)

        print(f'Epoch {epoch + 1}/{opt.n_epochs}, train loss: {train_loss:.4f}, train_vif: {train_vif_all:.4f}, '
              f'train_en: {train_en_all:.4f}, train_ssim: {train_ssim_all:.4f}, train_sf: {train_sf_all:.4f}, '
              f'train_q_abf: {train_q_abf_all:.4f}, train_ag: {train_ag_all:.4f}')
        print(f'test_vif: {test_vif_all:.4f}, test_en: {test_en_all:.4f}, test_ssim: {test_ssim_all:.4f}, '
              f'test_sf: {test_sf_all:.4f}, test_q_abf: {test_q_abf_all:.4f}, test_ag: {test_ag_all:.4f}')

        folder_path = os.path.join("model_train", f"{opt.model_sel}_{opt.tasks}_{opt.prior_sel}_{opt.int_style}")
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
