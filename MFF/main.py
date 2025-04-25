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
from JointAE_Blocks import JointAE_Blocks
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from LossFunction import M3Loss
from metrics import rmse, psnr, ssim, SF
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
from PIL import Image
import torchvision.transforms as transforms


def modal_selection(dataloader, device):
    ssim_A = 0.0
    ssim_B = 0.0
    ssim_AB = 0.0
    ssim_wavelet = 0.0
    ssim_laplacian = 0.0
    ssim_nsct = 0.0
    ssim_energy = 0.0
    ssim_gradient = 0.0
    ssim_feature = 0.0
    ssim_low_rank = 0.0
    ssim_guide = 0.0
    for batch_idx, (xA, xB, labels) in enumerate(dataloader):
        xA = xA.to(device)
        xB = xB.to(device)
        labels = labels.to(device)

        xAB = (xA + xB) / 2
        x_wavelet = wavelet_fusion(xA, xB, fusion_method='average')
        x_laplacian = laplacian_pyramid_fusion(xA, xB)
        x_nsct = nsct_fusion(xA, xB)
        x_energy = energy_minimization_fusion(xA, xB)
        x_gradient = gradient_field_fusion(xA, xB)
        x_feature = feature_level_fusion(xA, xB)
        x_low_rank = low_rank_matrix_decomposition_fusion(xA, xB)
        x_guide = guided_filter_fusion(xA, xB)

        ssim_A += ssim(xA, labels)
        ssim_B += ssim(xB, labels)
        ssim_AB += ssim(xAB, labels)
        ssim_wavelet += ssim(x_wavelet, labels)
        ssim_laplacian += ssim(x_laplacian, labels)
        ssim_nsct += ssim(x_nsct, labels)
        ssim_energy += ssim(x_energy, labels)
        ssim_gradient += ssim(x_gradient, labels)
        ssim_feature += ssim(x_feature, labels)
        ssim_low_rank += ssim(x_low_rank, labels)
        ssim_guide += ssim(x_guide, labels)
        break
    print(
        f'ssim:{ssim_A, ssim_B, ssim_AB, ssim_wavelet, ssim_laplacian, ssim_nsct, ssim_energy, ssim_gradient, ssim_feature, ssim_low_rank, ssim_guide}')
    sort_dict = {ssim_A: 'xA', ssim_B: 'xB', ssim_AB: 'xAB', ssim_wavelet: 'x_wav', ssim_laplacian: 'x_lap',
                 ssim_nsct: 'x_nsct', ssim_energy: 'x_ey', ssim_gradient: 'x_gra', ssim_feature: 'x_fea',
                 ssim_low_rank: 'x_lr', ssim_guide: 'x_gui'}

    sorted_dict = dict(sorted(sort_dict.items(), key=lambda item: item[0], reverse=True))
    for key, value in sorted_dict.items():
        print(f"{key}: {value}")

    max_key = max(sort_dict)

    return sort_dict[max_key]


# def model_test(init_style, epoch, model, dataloader_test, device, save_dir='image/'):
#     model.eval()
#     test_mse = 0.0
#     test_psnr = 0.0
#     test_ssim = 0.0
#     test_sf = 0.0
#     to_pil = transforms.ToPILImage()
#
#     save_images = (epoch % 1 == 0)
#
#     if save_images:
#         init_folder = os.path.join(save_dir, init_style)
#         epoch_folder = os.path.join(init_folder, f'epoch_{epoch}')
#         os.makedirs(epoch_folder, exist_ok=True)
#
#     with torch.no_grad():
#         for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_test):
#             near_focus = near_focus.to(device)
#             far_focus = far_focus.to(device)
#             labels = labels.to(device)
#
#             output = model(near_focus, far_focus)
#
#             test_mse += mse(output, labels)
#             test_psnr += psnr(output, labels)
#             test_ssim += ssim(output, labels)
#             test_sf += SF(output)
#
#             if save_images:
#                 output_image = to_pil(output.squeeze(0).cpu())
#                 output_image.save(os.path.join(epoch_folder, f'output_{batch_idx}.png'))
#
#         test_mse_all = test_mse / len(dataloader_test)
#         test_psnr_all = test_psnr / len(dataloader_test)
#         test_ssim_all = test_ssim / len(dataloader_test)
#         test_sf_all = test_sf / len(dataloader_test)
#
#     return test_mse_all, test_psnr_all, test_ssim_all, test_sf_all


def model_test(model, dataloader_test, device):
    model.eval()
    test_rmse = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_sf = 0.0
    with torch.no_grad():
        for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_test):
            near_focus = near_focus.to(device)
            far_focus = far_focus.to(device)
            labels = labels.to(device)

            output = model(near_focus, far_focus)

            test_rmse += rmse(output, labels)
            test_psnr += psnr(output, labels)
            test_ssim += ssim(output, labels)
            test_sf += SF(output)

        test_rmse_all = test_rmse / len(dataloader_test)
        test_psnr_all = test_psnr / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)
    return test_rmse_all, test_psnr_all, test_ssim_all, test_sf_all


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
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--loss", type=str, default='MSE', help="MSE or MAE or CIL")
    parser.add_argument("--if_blocks", type=str, default='no', help="yes or no")
    parser.add_argument("--input_size", type=int, default=3, help="")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--AE_num_layers", type=int, default=4, help="")
    parser.add_argument("--AE_num_blocks", type=int, default=1, help="")
    parser.add_argument("--pool_size", type=int, default=64, help="")
    parser.add_argument("--fs", type=float, default=0.2, help="")
    parser.add_argument("--threshold", type=float, default=0.05, help="")
    parser.add_argument("--gamma", type=float, default=0.7, help="")
    parser.add_argument("--l_int", type=float, default=1.0, help="")
    parser.add_argument("--l_ncc", type=float, default=1.0, help="")
    parser.add_argument("--l_ssim", type=float, default=1.0, help="")
    parser.add_argument("--l_grad", type=float, default=1.0, help="")
    parser.add_argument("--optim", type=str, default='AdamW', help="Adam or AdamW or SGD")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lr", type=float, default=0.0005, help="")
    parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="")
    parser.add_argument("--patience", type=int, default=30, help="")
    parser.add_argument("--model_sel", type=str, default='DRDB', help="RB or DB or RDB or ARDB or DRDB")
    parser.add_argument("--prior_sel", type=str, default='xB', help="")
    parser.add_argument("--model_path", type=str, default='model_train/model',
                        help="")
    opt = parser.parse_args()

    # dataset, dataset_test = dataloader(high=opt.high, width=opt.width)
    dataset, dataset_test, _ = dataloader()
    dataloader_train = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    # dataloader_modal_sel = DataLoader(dataset, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # prior_sel = modal_selection(dataloader_modal_sel, device)
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
        if opt.if_blocks == 'yes':
            model = JointAE_Blocks(input_size=opt.input_size, hidden_size=opt.hidden_size,
                                   AE_num_layers=opt.AE_num_layers, AE_num_blocks=opt.AE_num_blocks,
                                   pool_size=opt.pool_size, modal_sel=opt.prior_sel).to(device)
        else:
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

    test_mse_all, test_psnr_all, test_ssim_all, test_sf_all = model_test(model, dataloader_test, device)
    print(f'init test mse:{test_mse_all:.4f}, init test psnr:{test_psnr_all:.4f}, init test ssim:{test_ssim_all:.4f}, '
          f'init test sf:{test_sf_all:.4f}')

    criterion = M3Loss(fs=opt.fs, loss=opt.loss, threshold=opt.threshold, gamma=opt.gamma, l_int=opt.l_int,
                       l_ncc=opt.l_ncc, l_ssim=opt.l_ssim, l_grad=opt.l_grad).to(device)

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

    best_test_ssim = -np.inf
    counter = 0
    for epoch in range(opt.n_epochs):
        model.train()
        running_loss = 0.0
        train_rmse = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        train_sf = 0.0
        progress_bar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
        for batch_idx, (near_focus, far_focus, labels) in enumerate(dataloader_train):
            near_focus = near_focus.to(device)
            far_focus = far_focus.to(device)
            labels = labels.to(device)

            output = model(near_focus, far_focus)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_rmse += rmse(output, labels)
            train_psnr += psnr(output, labels)
            train_ssim += ssim(output, labels)
            train_sf += SF(output)

            running_loss += loss.item()
            progress_bar_train.update(1)
            progress_bar_train.set_postfix(loss=running_loss / len(dataloader_train))
        train_loss = running_loss / len(dataloader_train)
        train_rmse_all = train_rmse / len(dataloader_train)
        train_psnr_all = train_psnr / len(dataloader_train)
        train_ssim_all = train_ssim / len(dataloader_train)
        train_sf_all = train_sf / len(dataloader_train)
        progress_bar_train.close()

        test_rmse_all, test_psnr_all, test_ssim_all, test_sf_all = model_test(model, dataloader_test, device)

        print(f'Epoch {epoch + 1}/{opt.n_epochs}, train loss: {train_loss:.4f}, train_mse: {train_rmse_all:.4f}, '
              f'train_psnr: {train_psnr_all:.4f}, train_ssim: {train_ssim_all:.4f}, train_sf: {train_sf_all:.4f}, '
              f'test_rmse: {test_rmse_all:.4f}, test_psnr: {test_psnr_all:.4f}, test_ssim: {test_ssim_all:.4f}, test_sf: '
              f'{test_sf_all:.4f}')

        if best_test_ssim <= test_ssim_all:
            best_test_ssim = test_ssim_all
            print(best_test_ssim)
            torch.save(model.state_dict(), f"{opt.model_path}_{epoch + 1}")
            counter = 0
        else:
            counter += 1

        if counter >= opt.patience:
            print("Early stopping")
            break

    print(best_test_ssim)
    print("训练完成！")


if __name__ == "__main__":
    train()
