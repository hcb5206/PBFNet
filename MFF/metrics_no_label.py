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


def ssim(fusion, img1, img2, max_val: float = 1.0, win_size: int = 7) -> float:
    # up [0-1]
    fusion_np = fusion.detach().cpu().numpy()
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    ssim_values_1 = [
        structural_similarity(fusion_np[i], img1_np[i], data_range=max_val, win_size=win_size, channel_axis=0)
        for i in range(fusion_np.shape[0])
    ]
    ssim_values_2 = [
        structural_similarity(fusion_np[i], img2_np[i], data_range=max_val, win_size=win_size, channel_axis=0)
        for i in range(fusion_np.shape[0])
    ]

    return float(np.mean(ssim_values_1)) + float(np.mean(ssim_values_2))


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


def VIF(fusion, img1, img2):
    # up
    fusion_clipped = torch.clamp(fusion, 0, 1)
    img1_clipped = torch.clamp(img1, 0, 1)
    img2_clipped = torch.clamp(img2, 0, 1)
    return float(piq.vif_p(fusion_clipped, img1_clipped, data_range=1.0)) + float(
        piq.vif_p(fusion_clipped, img2_clipped, data_range=1.0))


def MI(fusion, img1, img2) -> float:
    # up
    fusion_np = fusion.argmax(dim=1).cpu().numpy()
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    nmi_scores = []

    for i in range(fusion_np.shape[0]):
        nmi = normalized_mutual_information(img1_np[i].flatten(),
                                            fusion_np[i].flatten()) + normalized_mutual_information(
            img2_np[i].flatten(), fusion_np[i].flatten())
        nmi_scores.append(nmi)

    return float(np.sum(nmi_scores))


def SD(images):
    # up
    std_devs = []
    for img in images:
        std_dev = torch.std(img)
        std_devs.append(std_dev.item())
    return float(torch.tensor(std_devs))


def compute_entropy(image):
    flattened = image.reshape(image.size(0), -1)
    entropy_total = 0.0

    for channel in range(flattened.size(0)):
        hist = torch.histc(flattened[channel], bins=256, min=0, max=1)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = -torch.sum(prob * torch.log2(prob))
        entropy_total += entropy

    return entropy_total.item() / flattened.size(0)


def EN(fusion):
    en_F = compute_entropy(fusion)
    return float(en_F)


def AG(image: torch.Tensor):
    image = image.squeeze(0).float()
    gradient_h = image[:, :, 1:] - image[:, :, :-1]
    gradient_v = image[:, 1:, :] - image[:, :-1, :]
    gradient_magnitude = torch.sqrt(gradient_h[:, :-1, :] ** 2 + gradient_v[:, :, :-1] ** 2)
    avg_gradient = gradient_magnitude.mean()
    return float(avg_gradient)


def SF(output: torch.Tensor) -> float:
    diff_h = torch.diff(output, dim=-1)
    diff_v = torch.diff(output, dim=-2)

    hf = torch.sqrt(torch.mean(diff_h ** 2, dim=(-2, -1)))
    vf = torch.sqrt(torch.mean(diff_v ** 2, dim=(-2, -1)))

    sf_per_channel = torch.sqrt(hf ** 2 + vf ** 2)

    sf_per_sample = torch.mean(sf_per_channel, dim=-1)

    sf_total = torch.sum(sf_per_sample).item()

    return float(sf_total)


# def SF(output: torch.Tensor) -> float:
#     output = output.float()
#     sf_per_sample = []
#     for sample in range(output.shape[0]):
#         sf_per_channel = []
#         for channel in range(output.shape[1]):
#             diff_h = (output[sample, channel, :, 1:] - output[sample, channel, :, :-1]) ** 2
#             hf = torch.sqrt(diff_h.mean())
#             diff_v = (output[sample, channel, 1:, :] - output[sample, channel, :-1, :]) ** 2
#             vf = torch.sqrt(diff_v.mean())
#             sf = torch.sqrt(hf ** 2 + vf ** 2)
#             sf_per_channel.append(sf.item())
#         sf_per_sample.append(float(torch.tensor(sf_per_channel).mean()))
#
#     return float(sum(sf_per_sample))


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


def q_abf(fusion, A, B):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

    sobel_x = sobel_x.repeat(3, 1, 1, 1)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)

    def compute_gradient(img):
        grad_x = F.conv2d(img, sobel_x, padding=1, groups=3).squeeze(0)
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=3).squeeze(0)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2)

    grad_A = compute_gradient(A)
    grad_B = compute_gradient(B)
    grad_F = compute_gradient(fusion)

    Q_A = (grad_F * grad_A).sum(dim=(1, 2)) / (
            torch.sqrt((grad_F ** 2).sum(dim=(1, 2))) * torch.sqrt((grad_A ** 2).sum(dim=(1, 2))) + 1e-8)
    Q_B = (grad_F * grad_B).sum(dim=(1, 2)) / (
            torch.sqrt((grad_F ** 2).sum(dim=(1, 2))) * torch.sqrt((grad_B ** 2).sum(dim=(1, 2))) + 1e-8)

    w_A = grad_A.sum(dim=(1, 2))
    w_B = grad_B.sum(dim=(1, 2))

    Q_ABF = (w_A * Q_A + w_B * Q_B).sum() / (w_A.sum() + w_B.sum() + 1e-8)

    return Q_ABF.item()

# if __name__ == '__main__':
#     fusion = torch.rand((1, 3, 256, 256))
#     img1 = torch.rand((1, 3, 256, 256))
#     img2 = torch.rand((1, 3, 256, 256))
#
#     ssim_value = ssim(fusion, img1, img2)
#     vif_value = VIF(fusion, img1, img2)
#     en_value = EN(fusion)
#     sf_value = SF(fusion)
#     Q_abf = q_abf(fusion, img1, img2)
#
#     print(f"ssim: {ssim_value:.4f}")
#     print(f"vif: {vif_value:.4f}")
#     print(f"en: {en_value:.4f}")
#     print(f"sf: {sf_value:.4f}")
#     print(f"Q_abf: {Q_abf:.4f}")
