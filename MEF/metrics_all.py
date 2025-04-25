import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error, \
    normalized_mutual_information
from math import exp
from skimage.measure import shannon_entropy
from skimage.feature import canny
from skimage import color
import piq


def mse(output: torch.Tensor, target: torch.Tensor) -> float:
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    mse_values = [mean_squared_error(target_np[i], output_np[i]) for i in range(output_np.shape[0])]

    return float(np.mean(mse_values))


def rmse(output: torch.Tensor, target: torch.Tensor) -> float:
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    rmse_values = [np.sqrt(mean_squared_error(target_np[i], output_np[i])) for i in range(output_np.shape[0])]

    return float(np.sum(rmse_values))


def psnr(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    # up
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    psnr_values = [peak_signal_noise_ratio(target_np[i], output_np[i], data_range=max_val) for i in
                   range(output_np.shape[0])]

    return float(np.mean(psnr_values))


def ssim(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, win_size: int = 7) -> float:
    # up [0-1]
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_values = [
        structural_similarity(target_np[i], output_np[i], data_range=max_val, win_size=win_size, channel_axis=0)
        for i in range(output_np.shape[0])
    ]

    return float(np.mean(ssim_values))


def gaussian_window(window_size: int, sigma: float):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int):
    _1D_window = gaussian_window(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_s(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])


def ms_ssim(img1, img2, window_size=11, size_average=True, weights=None):
    # up [0-1]
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    levels = len(weights)
    msssim = []

    for _ in range(levels):
        ssim_val = ssim_s(img1, img2, window, window_size, channel, size_average)
        msssim.append(ssim_val)

        img1 = F.avg_pool2d(img1, kernel_size=2)
        img2 = F.avg_pool2d(img2, kernel_size=2)

    msssim = torch.stack(msssim)
    ms_ssim_val = torch.prod(msssim ** torch.tensor(weights).to(img1.device))

    return float(ms_ssim_val)


def EPI(images1, images2):
    # up
    edge_indices = []

    for img1, img2 in zip(images1, images2):
        img1_gray = img1.mean(0)
        img2_gray = img2.mean(0)

        grad_x1 = torch.abs(img1_gray[:, :-1] - img1_gray[:, 1:])
        grad_y1 = torch.abs(img1_gray[:-1, :] - img1_gray[1:, :])
        grad_x2 = torch.abs(img2_gray[:, :-1] - img2_gray[:, 1:])
        grad_y2 = torch.abs(img2_gray[:-1, :] - img2_gray[1:, :])

        edges1 = (grad_x1.sum() + grad_y1.sum()) / (img1_gray.shape[0] * img1_gray.shape[1] - 1)
        edges2 = (grad_x2.sum() + grad_y2.sum()) / (img2_gray.shape[0] * img2_gray.shape[1] - 1)

        epi = edges2 / (edges1 + 1e-8)
        edge_indices.append(epi.item())

    return float(torch.tensor(edge_indices))


def VIF(img, ref):
    # up
    img_clipped = torch.clamp(img, 0, 1)
    ref_clipped = torch.clamp(ref, 0, 1)
    return float(piq.vif_p(img_clipped, ref_clipped, data_range=1.0))


def Q_MI(fused_output: torch.Tensor, true_labels: torch.Tensor) -> float:
    # up
    fused_output_np = fused_output.argmax(dim=1).cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()

    nmi_scores = []

    for i in range(fused_output_np.shape[0]):
        nmi = normalized_mutual_information(true_labels_np[i].flatten(), fused_output_np[i].flatten())
        nmi_scores.append(nmi)

    return float(np.sum(nmi_scores))


def SD(images):
    # up
    std_devs = []
    for img in images:
        std_dev = torch.std(img)
        std_devs.append(std_dev.item())
    return float(torch.tensor(std_devs))


def EN(images):
    # up
    batch_entropy = []
    for img in images:
        img_np = img.cpu().numpy()
        entropy_value = shannon_entropy(img_np, base=2)
        batch_entropy.append(entropy_value)
    return float(torch.tensor(batch_entropy))


def AG(images):
    # up
    gradients = []
    for img in images:
        img = img.mean(0)
        grad_x = torch.abs(img[:, :-1] - img[:, 1:])
        grad_y = torch.abs(img[:-1, :] - img[1:, :])
        avg_grad = (grad_x.mean() + grad_y.mean()) / 2
        gradients.append(avg_grad.item())
    return float(torch.tensor(gradients))


def SF(output: torch.Tensor) -> float:
    # up
    output_np = output.detach().cpu().numpy()

    sf_per_sample = []
    for sample in range(output_np.shape[0]):
        sf_per_channel = []
        for channel in range(output_np.shape[1]):
            diff_h = np.diff(output_np[sample, channel], axis=1) ** 2
            hf = np.sqrt(np.mean(diff_h))

            diff_v = np.diff(output_np[sample, channel], axis=0) ** 2
            vf = np.sqrt(np.mean(diff_v))

            sf = np.sqrt(hf ** 2 + vf ** 2)
            sf_per_channel.append(sf)

        sf_per_sample.append(float(np.mean(sf_per_channel)))

    return float(np.sum(sf_per_sample))


def calculate_entropy(image):
    histogram = torch.histc(image.flatten(), bins=256, min=0, max=255)
    histogram = histogram / histogram.sum()
    histogram = histogram[histogram > 0]
    entropy = -torch.sum(histogram * torch.log2(histogram))
    return entropy.item()


def Q_NICE(images):
    # up
    q_ncie_values = []

    for i in range(images.shape[0]):
        image = images[i].float()

        entropy_values = []
        for channel in range(image.shape[0]):
            entropy = calculate_entropy(image[channel])
            entropy_values.append(entropy)

        q_ncie = np.mean(entropy_values)
        q_ncie_values.append(q_ncie)

    return float(torch.tensor(q_ncie_values))

# output = torch.rand((4, 3, 256, 256))
# target = torch.rand((4, 3, 256, 256))
#
# mse_value = mse(output, target)
# psnr_value = psnr(output, target)
# ssim_value = ssim(output, target)
# sf_value = spatial_frequency(output)
# ms_ssim_value = ms_ssim(output, target)
#
# print(f"MSE: {mse_value:.4f}")
# print(f"PSNR: {psnr_value:.4f} dB")
# print(f"SSIM: {ssim_value:.4f}")
# print(f"Spatial Frequency: {sf_value:.4f}")
# print(f"MS SSIM: {ms_ssim_value:.4f}")
