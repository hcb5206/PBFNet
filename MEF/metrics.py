import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


def mse(output: torch.Tensor, target: torch.Tensor) -> float:
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    mse_values = [mean_squared_error(target_np[i], output_np[i]) for i in range(output_np.shape[0])]

    return float(np.mean(mse_values))


def psnr(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    psnr_values = [peak_signal_noise_ratio(target_np[i], output_np[i], data_range=max_val) for i in
                   range(output_np.shape[0])]

    return float(np.mean(psnr_values))


# def mef_ssim(fused_image: torch.Tensor, source_images: torch.Tensor, max_val: float = 1.0, win_size: int = 7) -> float:
#     fused_image_np = fused_image.detach().cpu().numpy()
#     source_images_np = source_images.detach().cpu().numpy()
#
#     batch_size, num_exposures, C, H, W = source_images_np.shape
#     mef_ssim_values = []
#
#     for i in range(batch_size):
#         ssim_sum = 0.0
#         for j in range(num_exposures):
#             ssim_value = structural_similarity(
#                 source_images_np[i, j], fused_image_np[i],
#                 data_range=max_val, win_size=win_size, channel_axis=0
#             )
#             ssim_sum += ssim_value
#
#         mef_ssim_score = ssim_sum / num_exposures
#         mef_ssim_values.append(mef_ssim_score)
#
#     return float(np.mean(mef_ssim_values))


def ssim(output: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, win_size: int = 7) -> float:
    output_np = output.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_values = [
        structural_similarity(target_np[i], output_np[i], data_range=max_val, win_size=win_size, channel_axis=0)
        for i in range(output_np.shape[0])
    ]

    return float(np.mean(ssim_values))


def spatial_frequency(output: torch.Tensor) -> float:
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

    return float(np.mean(sf_per_sample))

# output = torch.rand((4, 3, 256, 256))
# target = torch.rand((4, 3, 256, 256))
#
# mse_value = mse(output, target)
# psnr_value = psnr(output, target)
# ssim_value = ssim(output, target)
# sf_value = spatial_frequency(output)
#
# print(f"MSE: {mse_value:.4f}")
# print(f"PSNR: {psnr_value:.4f} dB")
# print(f"SSIM: {ssim_value:.4f}")
# print(f"Spatial Frequency: {sf_value:.4f}")
