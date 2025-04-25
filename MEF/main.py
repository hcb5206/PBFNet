import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
import csv
from torchvision.utils import save_image
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import dataloader
from JointAE import JointAE
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from LossFunction import M3Loss
from metrics import mse, psnr, ssim, spatial_frequency
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion


def visualize_and_save_batch(batch_idx, underexposed, overexposed, labels, save_dir="image"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(underexposed.cpu().squeeze().permute(1, 2, 0))
    axes[0].set_title("Underexposed")
    axes[0].axis("off")

    axes[1].imshow(overexposed.cpu().squeeze().permute(1, 2, 0))
    axes[1].set_title("Overexposed")
    axes[1].axis("off")

    axes[2].imshow(labels.cpu().squeeze().permute(1, 2, 0))
    axes[2].set_title("Label")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"batch_{batch_idx + 1}.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Batch {batch_idx} visualization saved at {save_path}")


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
        print(batch_idx)
        xA = xA.to(device)
        xB = xB.to(device)
        labels = labels.to(device)

        # source_images = torch.cat((xA.unsqueeze(dim=1), xB.unsqueeze(dim=1)), dim=1)
        # print(source_images.shape)

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

    ssim_A = ssim_A / len(dataloader)
    ssim_B = ssim_B / len(dataloader)
    ssim_AB = ssim_AB / len(dataloader)
    ssim_wavelet = ssim_wavelet / len(dataloader)
    ssim_laplacian = ssim_laplacian / len(dataloader)
    ssim_nsct = ssim_nsct / len(dataloader)
    ssim_energy = ssim_energy / len(dataloader)
    ssim_gradient = ssim_gradient / len(dataloader)
    ssim_feature = ssim_feature / len(dataloader)
    ssim_low_rank = ssim_low_rank / len(dataloader)
    ssim_guide = ssim_guide / len(dataloader)
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


# def model_test(model, dataloader_test, device):
#     model.eval()
#     test_mse = 0.0
#     test_psnr = 0.0
#     test_ssim = 0.0
#     test_sf = 0.0
#     with torch.no_grad():
#         for batch_idx, (underexposed, overexposed, labels) in enumerate(dataloader_test):
#             underexposed = underexposed.to(device)
#             overexposed = overexposed.to(device)
#             labels = labels.to(device)
#
#             output = model(underexposed, overexposed)
#
#             test_mse += mse(output, labels)
#             test_psnr += psnr(output, labels)
#             test_ssim += ssim(output, labels)
#             test_sf += spatial_frequency(output)
#
#         test_mse_all = test_mse / len(dataloader_test)
#         test_psnr_all = test_psnr / len(dataloader_test)
#         test_ssim_all = test_ssim / len(dataloader_test)
#         test_sf_all = test_sf / len(dataloader_test)
#     return test_mse_all, test_psnr_all, test_ssim_all, test_sf_all

def model_test(model, dataloader_test, device, epoch, csv_file='test_ssim_no_2.csv', output_dir='test_outputs_no_2'):
    model.eval()
    test_mse = 0.0
    test_psnr = 0.0
    test_ssim = 0.0
    test_sf = 0.0

    # epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    # os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (underexposed, overexposed, labels) in enumerate(dataloader_test):
            underexposed = underexposed.to(device)
            overexposed = overexposed.to(device)
            labels = labels.to(device)

            output = model(underexposed, overexposed)

            test_mse += mse(output, labels)
            test_psnr += psnr(output, labels)
            test_ssim += ssim(output, labels)
            test_sf += spatial_frequency(output)

            # output_img = output.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            # output_img = np.clip(output_img, 0, 1)
            # output_img = (output_img * 255).astype('uint8')
            # output_pil = Image.fromarray(output_img)
            # output_pil.save(os.path.join(epoch_dir, f"{batch_idx + 1:02d}.PNG"))

        test_mse_all = test_mse / len(dataloader_test)
        test_psnr_all = test_psnr / len(dataloader_test)
        test_ssim_all = test_ssim / len(dataloader_test)
        test_sf_all = test_sf / len(dataloader_test)

        # file_exists = os.path.isfile(csv_file)
        # with open(csv_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     if not file_exists:
        #         writer.writerow(['epoch', 'ssim'])
        #     writer.writerow([epoch + 1, test_ssim_all])

    return test_mse_all, test_psnr_all, test_ssim_all, test_sf_all


# def model_test(epoch, model, dataloader_test, device, csv_file_path="test_ssim_AB.csv", output_dir="test_images_AB"):
#     model.eval()
#     test_mse = 0.0
#     test_psnr = 0.0
#     test_ssim = 0.0
#     test_sf = 0.0
#
#     # Ensure the output directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # Create a subfolder for the current epoch
#     epoch_folder = os.path.join(output_dir, f"epoch_{epoch + 1}")
#     if not os.path.exists(epoch_folder):
#         os.makedirs(epoch_folder)
#
#     with torch.no_grad():
#         for batch_idx, (underexposed, overexposed, labels) in enumerate(dataloader_test):
#             underexposed = underexposed.to(device)
#             overexposed = overexposed.to(device)
#             labels = labels.to(device)
#
#             output = model(underexposed, overexposed)
#
#             test_mse += mse(output, labels)
#             test_psnr += psnr(output, labels)
#             test_ssim += ssim(output, labels)
#             test_sf += spatial_frequency(output)
#
#             # Save the test images (e.g., underexposed, overexposed, output, labels) to the epoch folder
#             for idx in range(underexposed.size(0)):
#                 # Save each image in the batch as a separate file
#                 save_image(underexposed[idx], os.path.join(epoch_folder, f"underexposed_{batch_idx}_{idx}.png"))
#                 save_image(overexposed[idx], os.path.join(epoch_folder, f"overexposed_{batch_idx}_{idx}.png"))
#                 save_image(labels[idx], os.path.join(epoch_folder, f"label_{batch_idx}_{idx}.png"))
#                 save_image(output[idx], os.path.join(epoch_folder, f"output_{batch_idx}_{idx}.png"))
#
#         test_mse_all = test_mse / len(dataloader_test)
#         test_psnr_all = test_psnr / len(dataloader_test)
#         test_ssim_all = test_ssim / len(dataloader_test)
#         test_sf_all = test_sf / len(dataloader_test)
#
#     # Save epoch and test_ssim_all to CSV
#     with open(csv_file_path, mode='a', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         if csv_file.tell() == 0:
#             writer.writerow(["epoch", "test_ssim_all"])
#         writer.writerow([epoch, test_ssim_all])
#
#     return test_mse_all, test_psnr_all, test_ssim_all, test_sf_all


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
    parser.add_argument("--loss", type=str, default='MSE', help="MSE or MAE or CIL")
    parser.add_argument("--input_size", type=int, default=3, help="")
    parser.add_argument("--hidden_size", type=int, default=64, help="")
    parser.add_argument("--AE_num_layers", type=int, default=4, help="")
    parser.add_argument("--pool_size", type=int, default=64, help="")
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
    parser.add_argument("--model_path", type=str, default='model_train/model',
                        help="")
    opt = parser.parse_args()

    dataset, dataset_test = dataloader()
    dataloader_train = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    # dataloader_modal_sel = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)

    # modal_sel = modal_selection(dataloader_modal_sel, device)
    # modal_sel = 'xA'
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

    criterion = M3Loss(fs=opt.fs, loss=opt.loss, threshold=opt.threshold, gamma=opt.gamma).to(device)

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
        train_mse = 0.0
        train_psnr = 0.0
        train_ssim = 0.0
        train_sf = 0.0
        progress_bar_train = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{opt.n_epochs}", unit="batch")
        for batch_idx, (underexposed, overexposed, labels) in enumerate(dataloader_train):
            underexposed = underexposed.to(device)
            overexposed = overexposed.to(device)
            labels = labels.to(device)

            # visualize_and_save_batch(batch_idx, underexposed, overexposed, labels, save_dir="image")

            output = model(underexposed, overexposed)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse += mse(output, labels)
            train_psnr += psnr(output, labels)
            train_ssim += ssim(output, labels)
            train_sf += spatial_frequency(output)

            running_loss += loss.item()
            progress_bar_train.update(1)
            progress_bar_train.set_postfix(loss=running_loss / len(dataloader_train))
        train_loss = running_loss / len(dataloader_train)
        train_mse_all = train_mse / len(dataloader_train)
        train_psnr_all = train_psnr / len(dataloader_train)
        train_ssim_all = train_ssim / len(dataloader_train)
        train_sf_all = train_sf / len(dataloader_train)
        progress_bar_train.close()

        # test_mse_all, test_psnr_all, test_ssim_all, test_sf_all = model_test(epoch, model, dataloader_test, device)
        test_mse_all, test_psnr_all, test_ssim_all, test_sf_all = model_test(model, dataloader_test, device, epoch)

        print(f'Epoch {epoch + 1}/{opt.n_epochs}, train loss: {train_loss:.4f}, train_mse: {train_mse_all:.4f}, '
              f'train_psnr: {train_psnr_all:.4f}, train_ssim: {train_ssim_all:.4f}, '
              f'train_sf: {train_sf_all:.4f}, test_mse: {test_mse_all:.4f}, test_psnr: {test_psnr_all:.4f}, '
              f'test_ssim: {test_ssim_all:.4f}, test_sf:{test_sf_all:.4f}')

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
