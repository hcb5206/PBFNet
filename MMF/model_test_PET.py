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
from dataloader import dataloader_MRI_PET, dataloader_MRI_SPECT
from JointAE import JointAE
from ResBlock import JointAE as JointAE_res
from DenseBlock import JointAE as JointAE_dense
from ResDenseBlock import JointAE as JointAE_res_dense
from AggResDenseBlock import JointAE as JointAE_agg_res_dense
from LossFunction import M3Loss
from metrics import ssim, VIF, EN, SF, q_abf, AG
from math import exp
import time
from thop import profile


def save_and_visualize_image(output, save_dir, img_idx):
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


# print(Residual_Metric(0.0065, 0.0198, 'rmse'))
# print(Residual_Metric(44.6042, 35.193))
# print(Residual_Metric(0.9929, 0.9498))
# print(Residual_Metric(0.9443, 0.8027))

# print(Residual_Metric(0.0712, 0.0544))
# print(Residual_Metric(0.2518, 0.2397))
# print(Residual_Metric(0.0252, 0.0185))


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

    # all_flops = 0.0
    # all_time = 0.0
    with torch.no_grad():
        for batch_idx, (mri, pet) in enumerate(dataloader_test):
            mri = mri.to(device)
            pet = pet.to(device)

            # start_time = time.time()
            output = model(mri, pet)
            # end_time = time.time()
            # res_time = end_time - start_time
            # all_time += res_time

            # flops, _ = profile(model, inputs=(mri, pet))
            # all_flops += flops / 1e9

            input_img = torch.maximum(mri, pet)

            input_vif += VIF(input_img, mri, pet)
            input_en += EN(input_img)
            input_ssim += ssim(input_img, mri, pet)
            input_sf += SF(input_img)
            input_q_abf += q_abf(input_img, mri, pet)
            input_ag += AG(input_img)

            test_vif += VIF(output, mri, pet)
            test_en += EN(output)
            test_ssim += ssim(output, mri, pet)
            test_sf += SF(output)
            test_q_abf += q_abf(output, mri, pet)
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

        # avg_flops = all_flops / len(dataloader_test)
        # avg_time = all_time / len(dataloader_test)

    print(f'test_vif:{test_vif_all:.4f}, test_en:{test_en_all:.4f}, test_ssim:{test_ssim_all:.4f}, '
          f'test_sf:{test_sf_all:.4f}, test_q_abf:{test_q_abf_all:.4f}, test_ag:{test_ag_all:.4f}')
    print(f'input_vif:{input_vif_all:.4f}, input_en:{input_en_all:.4f}, input_ssim:{input_ssim_all:.4f}, '
          f'input_sf:{input_sf_all:.4f}, input_q_abf:{input_q_abf_all:.4f}, input_ag:{input_ag_all:.4f}')
    print(f'test_vif:{test_vif_all:.4f}, res_vif:{res_vif:.4f}, test_en:{test_en_all:.4f}, '
          f'res_en:{res_en:.4f}, test_ssim:{test_ssim_all:.4f}, res_ssim:{res_ssim:.4f}, '
          f'test_sf:{test_sf_all:.4f}, res_sf:{res_sf:.4f}, test_q_abf:{test_q_abf_all:.4f}, res_q_abf:{res_q_abf:.4f}, '
          f'test_ag:{test_ag_all:.4f}, res_ag:{res_ag:.4f}')
    # print(f'avg_flops:{avg_flops:.4f}(G), avg_time:{avg_time:.4f}(s)')


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

    model.load_state_dict(torch.load(opt.model_path))
    print('The total number of parameters', count_parameters(model))
    model_test(model=model, dataloader_test=dataloader_test, device=device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (mri, pet) in enumerate(dataloader_test):
            mri = mri.to(device)
            pet = pet.to(device)

            output = model(mri, pet)
            # output = torch.maximum(mri, pet)

            # mri = normalize_image(mri)
            # pet = normalize_image(pet)
            # output = normalize_image(output)

            output_np = output.cpu().numpy().squeeze()
            output_np = output_np.transpose(1, 2, 0)

            save_and_visualize_image(output_np, 'image', img_idx=batch_idx + 1)


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
parser.add_argument("--lr", type=float, default=0.0001, help="")
parser.add_argument("--weight_decay_if", type=str, default='No', help="Yes or No")
parser.add_argument("--weight_decay", type=float, default=0.0005, help="")
parser.add_argument("--patience", type=int, default=30, help="")
parser.add_argument("--int_style", type=str, default='max',
                    help="max or or mean or mask_n or mask_g or no_ir or aa")
parser.add_argument("--prior_sel", type=str, default='x_max', help="")
parser.add_argument("--model_sel", type=str, default='DRDB', help="RB or DB or RDB or ARDB or DRDB")
parser.add_argument("--tasks", type=str, default='PET', help="PET or SPECT")
parser.add_argument("--model_path", type=str, default='model/PET/model.pth',
                    help="")
opt = parser.parse_args()
print(opt)

if opt.tasks == 'PET':
    _, dataset_test = dataloader_MRI_PET()
else:
    _, dataset_test = dataloader_MRI_SPECT()

model_test_visualization(dataset_test=dataset_test)
