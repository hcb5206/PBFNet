import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import piq


class CILLoss(nn.Module):
    def __init__(self, fs):
        super(CILLoss, self).__init__()
        self.fs = fs
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, predictions, targets):
        loss = self.fs * self.mse(predictions, targets) + (1 - self.fs) * self.mae(predictions, targets)
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = None

    @staticmethod
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    @staticmethod
    def _ssim(input, target, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(input, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(input * input, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(input * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, input, target):
        (_, channel, _, _) = input.size()

        if self.window is None or channel != self.channel:
            self.window = self.create_window(self.window_size, channel)
            self.window = self.window.to(input.device).type_as(input)
            self.channel = channel

        return 1 - self._ssim(input, target, self.window, self.window_size, channel, self.size_average)


class NCCLoss(nn.Module):
    def __init__(self, window_size=9):
        super(NCCLoss, self).__init__()
        self.window_size = window_size
        self.eps = 1e-8

    def forward(self, I1, I2):
        mean_I1 = self.sliding_window_mean(I1, self.window_size)
        mean_I2 = self.sliding_window_mean(I2, self.window_size)

        I1_zero_mean = I1 - mean_I1
        I2_zero_mean = I2 - mean_I2

        numerator = torch.sum(I1_zero_mean * I2_zero_mean, dim=[2, 3])

        denominator = torch.sqrt(torch.sum(I1_zero_mean ** 2, dim=[2, 3]) * torch.sum(I2_zero_mean ** 2, dim=[2, 3]))

        ncc = numerator / (denominator + self.eps)

        loss = 1 - torch.mean(ncc, dim=1)

        return torch.mean(loss)

    @staticmethod
    def sliding_window_mean(I, window_size):
        padding = window_size // 2
        kernel = torch.ones(I.shape[1], 1, window_size, window_size, device=I.device) / (window_size ** 2)

        mean = torch.nn.functional.conv2d(I, kernel, padding=padding, groups=I.shape[1])

        return mean


class MINELoss(nn.Module):
    def __init__(self, channels, height, width, patch_size):
        super(MINELoss, self).__init__()
        self.height = height // patch_size
        self.width = width // patch_size
        self.patch_size = patch_size
        self.flatten = nn.Flatten()
        flattened_dim = channels * self.height * self.width * 2
        self.layers = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1)
        )

    def forward(self, x, y):
        x = self.image_into_patches(x, self.patch_size).permute(0, 4, 1, 2, 3).reshape(-1, x.shape[1], self.height,
                                                                                       self.width)
        y = self.image_into_patches(y, self.patch_size).permute(0, 4, 1, 2, 3).reshape(-1, y.shape[1], self.height,
                                                                                       self.width)
        batch_size = x.size(0)
        x_flat = self.flatten(x)
        y_flat = self.flatten(y)

        tiled_x = torch.cat([x_flat, x_flat], dim=0)
        idx = torch.randperm(batch_size)
        shuffled_y = y_flat[idx]
        concat_y = torch.cat([y_flat, shuffled_y], dim=0)

        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = 1 - torch.tanh(torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        return loss

    @staticmethod
    def image_into_patches(image_tensor, patch_num):
        # input images shape: [batch_size, channels, height, width]
        # output images shape: [batch_size, channels, height / patch_num, width / patch_num, patch_num * patch_num]
        _, _, height, width = image_tensor.shape

        patches_h = height // patch_num
        patches_w = width // patch_num

        patches = []
        for h in range(patch_num):
            for w in range(patch_num):
                patches.append(image_tensor[:, :, h * patches_h:h * patches_h + patches_h,
                               w * patches_w:w * patches_w + patches_w])

        return torch.stack(patches, dim=4)


class MutualInformationLoss(nn.Module):
    def __init__(self, num_bins=256):
        super(MutualInformationLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, I_complementary, I_target):
        I_complementary_flat = I_complementary.view(-1)
        I_target_flat = I_target.view(-1)

        bin_idx_x = (I_complementary_flat * (self.num_bins - 1)).clamp(0, self.num_bins - 1).long()
        bin_idx_y = (I_target_flat * (self.num_bins - 1)).clamp(0, self.num_bins - 1).long()

        hist_2d = torch.zeros((self.num_bins, self.num_bins), device=I_complementary.device)
        hist_2d[bin_idx_x, bin_idx_y] += 1

        hist_2d = hist_2d / hist_2d.sum()

        p_x = hist_2d.sum(dim=1)
        p_y = hist_2d.sum(dim=0)

        non_zero_indices = torch.nonzero(hist_2d > 0, as_tuple=False)

        hist_values = hist_2d[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        p_x_values = p_x[non_zero_indices[:, 0]]
        p_y_values = p_y[non_zero_indices[:, 1]]

        mi = (hist_values * torch.log(hist_values / (p_x_values * p_y_values + 1e-10))).sum()

        return 1 - torch.tanh(mi)


class GradientLoss(nn.Module):
    def __init__(self, if_th, threshold, gamma, operator='sobel'):
        super(GradientLoss, self).__init__()
        self.if_th = if_th
        self.threshold = threshold
        self.gamma = gamma

        self.kernels = {
            'laplacian': torch.tensor([[0, 1, 0],
                                       [1, -4, 1],
                                       [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda(),
            'sobel_x': torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda(),
            'sobel_y': torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda(),
            'prewitt_x': torch.tensor([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda(),
            'prewitt_y': torch.tensor([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        }

        self.operator = operator

    def forward(self, pred, vi, ir):
        if self.operator == 'sobel':
            gradient_pred_x = F.conv2d(pred, self.kernels['sobel_x'], padding=1, groups=1)
            gradient_pred_y = F.conv2d(pred, self.kernels['sobel_y'], padding=1, groups=1)
            gradient_vi_x = F.conv2d(vi, self.kernels['sobel_x'], padding=1, groups=1)
            gradient_vi_y = F.conv2d(vi, self.kernels['sobel_y'], padding=1, groups=1)
            gradient_ir_x = F.conv2d(ir, self.kernels['sobel_x'], padding=1, groups=1)
            gradient_ir_y = F.conv2d(ir, self.kernels['sobel_y'], padding=1, groups=1)

            gradient_pred = torch.sqrt(gradient_pred_x ** 2 + gradient_pred_y ** 2 + 1e-10)
            gradient_vi = torch.sqrt(gradient_vi_x ** 2 + gradient_vi_y ** 2 + 1e-10)
            gradient_ir = torch.sqrt(gradient_ir_x ** 2 + gradient_ir_y ** 2 + 1e-10)

        elif self.operator == 'prewitt':
            gradient_pred_x = F.conv2d(pred, self.kernels['prewitt_x'], padding=1, groups=1)
            gradient_pred_y = F.conv2d(pred, self.kernels['prewitt_y'], padding=1, groups=1)
            gradient_vi_x = F.conv2d(vi, self.kernels['prewitt_x'], padding=1, groups=1)
            gradient_vi_y = F.conv2d(vi, self.kernels['prewitt_y'], padding=1, groups=1)
            gradient_ir_x = F.conv2d(ir, self.kernels['prewitt_x'], padding=1, groups=1)
            gradient_ir_y = F.conv2d(ir, self.kernels['prewitt_y'], padding=1, groups=1)

            gradient_pred = torch.sqrt(gradient_pred_x ** 2 + gradient_pred_y ** 2 + 1e-10)
            gradient_vi = torch.sqrt(gradient_vi_x ** 2 + gradient_vi_y ** 2 + 1e-10)
            gradient_ir = torch.sqrt(gradient_ir_x ** 2 + gradient_ir_y ** 2 + 1e-10)

        else:
            gradient_pred = F.conv2d(pred, self.kernels['laplacian'], padding=1, groups=1)
            gradient_vi = F.conv2d(vi, self.kernels['laplacian'], padding=1, groups=1)
            gradient_ir = F.conv2d(ir, self.kernels['laplacian'], padding=1, groups=1)

        gradient_pred = torch.abs(gradient_pred)
        gradient_vi = torch.abs(gradient_vi)
        gradient_ir = torch.abs(gradient_ir)

        # gradient_pred = (gradient_pred - gradient_pred.min()) / (
        #         gradient_pred.max() - gradient_pred.min() + 1e-10)
        # gradient_vi = (gradient_vi - gradient_vi.min()) / (
        #         gradient_vi.max() - gradient_vi.min() + 1e-10)
        # gradient_ir = (gradient_ir - gradient_ir.min()) / (
        #         gradient_ir.max() - gradient_ir.min() + 1e-10)

        if self.if_th == 'yes':
            gradient_pred_thresh = gradient_pred.clone()
            gradient_vi_thresh = gradient_vi.clone()
            gradient_ir_thresh = gradient_ir.clone()

            gradient_pred_thresh[gradient_pred_thresh > self.threshold] = 0
            gradient_vi_thresh[gradient_vi_thresh > self.threshold] = 0
            gradient_ir_thresh[gradient_ir_thresh > self.threshold] = 0

            gradient_pred = self.gamma_correction(gradient_pred_thresh, self.gamma)
            gradient_vi = self.gamma_correction(gradient_vi_thresh, self.gamma)
            gradient_ir = self.gamma_correction(gradient_ir_thresh, self.gamma)
        else:
            gradient_pred = self.gamma_correction(gradient_pred, self.gamma)
            gradient_vi = self.gamma_correction(gradient_vi, self.gamma)
            gradient_ir = self.gamma_correction(gradient_ir, self.gamma)

        loss = F.l1_loss(gradient_pred, torch.maximum(gradient_vi, gradient_ir))

        return loss

    @staticmethod
    def gamma_correction(image, gamma):
        return torch.pow(image + 1e-10, gamma)


def gamma_trans(image, gamma):
    return torch.pow(image + 1e-10, gamma)


def RGBYCrCb(rgb_image: torch.Tensor) -> torch.Tensor:
    if rgb_image.size(1) != 3:
        raise ValueError("输入张量的通道数必须为 3（RGB 图像）")

    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=rgb_image.dtype, device=rgb_image.device)

    rgb_image = rgb_image.permute(0, 2, 3, 1)

    y = torch.tensordot(rgb_image, transform_matrix[0], dims=([-1], [0]))

    y = y.unsqueeze(1)

    return y


def generate_mask(mri, pet) -> torch.Tensor:
    mask = (mri >= pet).float()
    return mask


class M3Loss(nn.Module):
    def __init__(self, fs, loss, if_th, threshold, gamma, l_int, l_ncc, l_ssim, l_grad, int_style, mask_th=0.3,
                 mask_gamma=0.8):
        super(M3Loss, self).__init__()
        self.loss = loss
        self.l_int = l_int
        self.l_ncc = l_ncc
        self.l_ssim = l_ssim
        self.l_grad = l_grad
        self.int_style = int_style
        self.mask_th = mask_th
        self.mask_gamma = mask_gamma
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cil = CILLoss(fs=fs)
        self.ncc = NCCLoss()
        self.mil = MutualInformationLoss()
        self.ssim = SSIMLoss()
        # self.ssim = piq.SSIMLoss()
        # self.ssim = piq.MultiScaleSSIMLoss()
        self.gradient = GradientLoss(if_th=if_th, threshold=threshold, gamma=gamma)

    def forward(self, output, vi, ir):
        # mask = generate_mask(vi, ir)
        if self.loss == 'MSE':
            if self.int_style == 'max':
                loss_fusion = self.mse(output, torch.maximum(vi, ir))
            elif self.int_style == 'mean':
                loss_fusion = self.mse(output, 0.55 * vi + (1 - 0.55) * ir)
            elif self.int_style == 'mask_n':
                loss_fusion = 0.5 * self.mse(output, ir) + 0.5 * self.mse(output, mask * vi)
            elif self.int_style == 'mask_g':
                loss_fusion = self.mse(output, torch.maximum(vi, gamma_trans(mask * ir, gamma=self.mask_gamma)))
            elif self.int_style == 'no_ir':
                loss_fusion = self.mse(output, vi)
            elif self.int_style == 'aa':
                loss_fusion = 0.5 * self.mse(output, vi) + 0.5 * self.mse(output, ir)
            else:
                raise ValueError(f"int_style must be either 'max' or 'mask_n' or 'mask_g' or 'mask_no'!")
        elif self.loss == 'MAE':
            if self.int_style == 'max':
                loss_fusion = self.mae(output, torch.maximum(vi, ir))
            elif self.int_style == 'mean':
                loss_fusion = self.mae(output, 0.55 * vi + (1 - 0.55) * ir)
            elif self.int_style == 'mask_n':
                loss_fusion = 0.5 * self.mae(output, ir) + 0.5 * self.mae(output, mask * vi)
            elif self.int_style == 'mask_g':
                loss_fusion = self.mae(output, torch.maximum(vi, gamma_trans(mask * ir, gamma=self.mask_gamma)))
            elif self.int_style == 'no_ir':
                loss_fusion = self.mae(output, vi)
            elif self.int_style == 'aa':
                loss_fusion = 0.5 * self.mae(output, vi) + 0.5 * self.mae(output, ir)
            else:
                raise ValueError(f"int_style must be either 'max' or 'mask_n' or 'mask_g' or 'mask_no'!")
        elif self.loss == 'CIL':
            if self.int_style == 'max':
                loss_fusion = self.cil(output, torch.maximum(vi, ir))
            elif self.int_style == 'mean':
                loss_fusion = self.cil(output, 0.55 * vi + (1 - 0.55) * ir)
            elif self.int_style == 'mask_n':
                loss_fusion = 0.5 * self.cil(output, ir) + 0.5 * self.cil(output, mask * vi)
            elif self.int_style == 'mask_g':
                loss_fusion = self.cil(output, torch.maximum(vi, gamma_trans(mask * ir, gamma=self.mask_gamma)))
            elif self.int_style == 'no_ir':
                loss_fusion = self.cil(output, vi)
            elif self.int_style == 'aa':
                loss_fusion = 0.5 * self.cil(output, vi) + 0.5 * self.cil(output, ir)
            else:
                raise ValueError(f"int_style must be either 'max' or 'mask_n' or 'mask_g' or 'mask_no'!")
        else:
            raise ValueError(f"loss must be either 'MSE' or 'MAE' or 'CIL'!")
        loss_fusion = self.l_int * loss_fusion
        loss_ncc = self.l_ncc * (0.5 * self.ncc(output, vi) + 0.5 * self.ncc(output, ir))
        loss_ssim = self.l_ssim * (0.5 * self.ssim(output, vi) + 0.5 * self.ssim(output, ir))
        loss_grad = self.l_grad * self.gradient(output, vi, ir)
        loss_output = loss_fusion + loss_ncc + loss_ssim + loss_grad
        # print(loss_fusion)
        # print(loss_ncc)
        # print(loss_ssim)
        # print(loss_grad)
        return loss_output
