import pywt
import torch
import cv2
import numpy as np
from skimage.transform import pyramid_expand
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter


def wavelet_fusion(img1, img2, wavelet='db1', fusion_method='average'):
    batch_size, channels, height, width = img1.shape
    fused_imgs = torch.zeros_like(img1)

    for b in range(batch_size):
        for c in range(channels):
            coeffs1 = pywt.dwt2(img1[b, c].cpu().numpy(), wavelet=wavelet)
            coeffs2 = pywt.dwt2(img2[b, c].cpu().numpy(), wavelet=wavelet)

            cA1, (cH1, cV1, cD1) = coeffs1
            cA2, (cH2, cV2, cD2) = coeffs2

            if fusion_method == 'max':
                cA_fused = np.maximum(cA1, cA2)
            elif fusion_method == 'average':
                cA_fused = (cA1 + cA2) / 2
            else:
                raise ValueError("Unsupported fusion method")

            cH_fused = np.maximum(cH1, cH2)
            cV_fused = np.maximum(cV1, cV2)
            cD_fused = np.maximum(cD1, cD2)

            fused_channel = pywt.idwt2((cA_fused, (cH_fused, cV_fused, cD_fused)), wavelet=wavelet)
            fused_imgs[b, c] = torch.tensor(fused_channel, dtype=img1.dtype, device=img1.device)

    return fused_imgs


def laplacian_pyramid_fusion(img1, img2, levels=3):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        batch_fused_channels = []
        for c in range(channels):
            img1_np = img1[b, c].cpu().numpy()
            img2_np = img2[b, c].cpu().numpy()

            lp_img1 = [img1_np]
            lp_img2 = [img2_np]
            for i in range(levels):
                img1_np = cv2.pyrDown(img1_np)
                img2_np = cv2.pyrDown(img2_np)
                lp_img1.append(img1_np)
                lp_img2.append(img2_np)

            fused_pyramid = []
            for i in range(levels):
                fused_pyramid.append(np.maximum(lp_img1[i], lp_img2[i]))

            fused_img = fused_pyramid[-1]
            for i in range(levels - 1, -1, -1):
                fused_img = cv2.pyrUp(fused_img)
                fused_img = cv2.resize(fused_img, (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))
                fused_img = cv2.add(fused_img, fused_pyramid[i])

            batch_fused_channels.append(torch.tensor(fused_img))

        fused_imgs.append(torch.stack(batch_fused_channels, dim=0))

    return torch.stack(fused_imgs, dim=0)


def nsct_fusion(img1, img2):
    batch_size, channels, height, width = img1.shape
    fused_imgs = torch.zeros_like(img1)

    for b in range(batch_size):
        for c in range(channels):
            img1_np = img1[b, c].cpu().numpy()
            img2_np = img2[b, c].cpu().numpy()

            high_frequencies_img1 = pyramid_expand(img1_np, upscale=2, channel_axis=None)
            high_frequencies_img2 = pyramid_expand(img2_np, upscale=2, channel_axis=None)

            high_freq_fused = np.maximum(high_frequencies_img1, high_frequencies_img2)

            fused_img = cv2.resize(high_freq_fused, (width, height))

            fused_imgs[b, c] = torch.tensor(fused_img)

    return fused_imgs


def energy_minimization_fusion(img1, img2):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        fused_channels = []
        for c in range(channels):
            img1_np = img1[b, c].cpu().numpy().astype(np.float32)
            img2_np = img2[b, c].cpu().numpy().astype(np.float32)

            if img1_np.dtype != np.float32:
                img1_np = img1_np.astype(np.float32)
            if img2_np.dtype != np.float32:
                img2_np = img2_np.astype(np.float32)

            grad_img1 = cv2.Laplacian(img1_np, cv2.CV_32F)
            grad_img2 = cv2.Laplacian(img2_np, cv2.CV_32F)

            fused_img = np.where(np.abs(grad_img1) > np.abs(grad_img2), img1_np, img2_np)
            fused_channels.append(torch.tensor(fused_img))

        fused_imgs.append(torch.stack(fused_channels, dim=0))

    return torch.stack(fused_imgs, dim=0)


def gradient_field_fusion(img1, img2):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        fused_channels = []
        for c in range(channels):
            img1_np = img1[b, c].cpu().numpy()
            img2_np = img2[b, c].cpu().numpy()

            grad_img1 = np.gradient(img1_np)
            grad_img2 = np.gradient(img2_np)

            fused_grad = [(g1 + g2) / 2 for g1, g2 in zip(grad_img1, grad_img2)]

            fused_img = denoise_tv_chambolle(np.mean(fused_grad, axis=0), weight=0.1)
            fused_channels.append(torch.tensor(fused_img))

        fused_imgs.append(torch.stack(fused_channels, dim=0))

    return torch.stack(fused_imgs, dim=0)


def feature_level_fusion(img1, img2):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        fused_channels = []
        for c in range(channels):
            img1_np = (img1[b, c].cpu().numpy() * 255).astype(np.uint8)
            img2_np = (img2[b, c].cpu().numpy() * 255).astype(np.uint8)

            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(img1_np, None)
            keypoints2, descriptors2 = orb.detectAndCompute(img2_np, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            matches = sorted(matches, key=lambda x: x.distance)

            fused_img = np.zeros_like(img1_np, dtype=np.float32)

            for match in matches:
                idx1 = match.queryIdx
                idx2 = match.trainIdx
                pt1 = tuple(map(int, keypoints1[idx1].pt))
                pt2 = tuple(map(int, keypoints2[idx2].pt))

                fused_img[pt1[1], pt1[0]] = max(img1_np[pt1[1], pt1[0]], img2_np[pt2[1], pt2[0]])

            mask = (fused_img > 0)
            fused_img[~mask] = (img1_np[~mask].astype(float) + img2_np[~mask].astype(float)) / 2

            fused_channels.append(torch.tensor(fused_img / 255.0, dtype=torch.float32))

        fused_imgs.append(torch.stack(fused_channels, dim=0))

    return torch.stack(fused_imgs, dim=0)


def low_rank_matrix_decomposition_fusion(img1, img2):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        img1_np = img1[b].cpu().numpy().transpose(1, 2, 0)
        img2_np = img2[b].cpu().numpy().transpose(1, 2, 0)

        img1_np = img1_np.astype(np.float32) / 255.0
        img2_np = img2_np.astype(np.float32) / 255.0

        U1, S1, V1 = np.linalg.svd(img1_np.reshape(-1, 3), full_matrices=False)
        U2, S2, V2 = np.linalg.svd(img2_np.reshape(-1, 3), full_matrices=False)

        rank = 1
        fused_singular_values = (S1[:rank] + S2[:rank]) / 2

        fused_img = np.dot(U1[:, :rank], np.diag(fused_singular_values)).dot(V1[:rank, :])
        fused_img = fused_img.reshape(height, width, channels)

        fused_img = np.clip(fused_img, 0, 1) * 255
        fused_img = fused_img.astype(np.uint8)

        fused_imgs.append(torch.tensor(fused_img).permute(2, 0, 1))

    return torch.stack(fused_imgs, dim=0)


def guided_filter(img1, img2, radius=3, eps=0.01):
    mean_I = gaussian_filter(img1, sigma=radius)
    mean_p = gaussian_filter(img2, sigma=radius)
    mean_Ip = gaussian_filter(img1 * img2, sigma=radius)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = gaussian_filter(img1 * img1, sigma=radius) - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = gaussian_filter(a, sigma=radius)
    mean_b = gaussian_filter(b, sigma=radius)

    return mean_a * img1 + mean_b


def guided_filter_fusion(img1, img2, radius=3, eps=0.01):
    batch_size, channels, height, width = img1.shape
    fused_imgs = []

    for b in range(batch_size):
        img1_np = (img1[b].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        img2_np = (img2[b].cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

        fused_img = guided_filter(img1_np, img2_np, radius, eps)

        fused_imgs.append(torch.tensor(fused_img.astype(np.float32) / 255.0).permute(2, 0, 1))

    return torch.stack(fused_imgs, dim=0)


# if __name__ == '__main__':
#     img_near_focus = torch.rand(4, 3, 424, 624, dtype=torch.float32, device='cuda')
#     img_far_focus = torch.rand(4, 3, 424, 624, dtype=torch.float32, device='cuda')
#
#     wavelet_result = wavelet_fusion(img_near_focus, img_far_focus, wavelet='db1', fusion_method='max')
#     laplacian_result = laplacian_pyramid_fusion(img_near_focus, img_far_focus, levels=3)
#     nsct_result = nsct_fusion(img_near_focus, img_far_focus)
#     energy_result = energy_minimization_fusion(img_near_focus, img_far_focus)
#     gradient_result = gradient_field_fusion(img_near_focus, img_far_focus)
#     feature_result = feature_level_fusion(img_near_focus, img_far_focus)
#     low_rank_result = low_rank_matrix_decomposition_fusion(img_near_focus, img_far_focus)
#     guide_result = guided_filter_fusion(img_near_focus, img_far_focus)
#     print(wavelet_result.shape, laplacian_result.shape, nsct_result.shape, energy_result.shape, gradient_result.shape,
#           feature_result.shape, low_rank_result.shape, guide_result.shape)
