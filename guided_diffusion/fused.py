import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MaxGradientLoss3D(nn.Module):
    def __init__(self, device='cuda'):
        super(MaxGradientLoss3D, self).__init__()
        # 定义 3D 差分算子 (针对 [B, C, D, H, W] 空间)
        # kernel_z 对应 Depth (D) 维度
        self.kernel_d = torch.tensor([[[[-1]], [[1]]]], dtype=torch.float32).view(1, 1, 2, 1, 1).to(device)
        # kernel_y 对应 Height (H) 维度
        self.kernel_h = torch.tensor([[[[-1], [1]]]], dtype=torch.float32).view(1, 1, 1, 2, 1).to(device)
        # kernel_x 对应 Width (W) 维度
        self.kernel_w = torch.tensor([[[[-1, 1]]]], dtype=torch.float32).view(1, 1, 1, 1, 2).to(device)

    def calc_gradient_3d(self, x):
        # 输入 x: [B, C, H, W, D]

        # 1. 维度调整: [B, C, H, W, D] -> [B, C, D, H, W]
        # 0:B, 1:C, 4:D, 2:H, 3:W
        x_p = x.permute(0, 1, 4, 2, 3)

        # 2. 针对 B*C 进行 3D 卷积 (使用 group conv 技巧或 reshape)
        b, c, d, h, w = x_p.shape
        x_reshaped = x_p.reshape(b * c, 1, d, h, w)

        # 3. 计算梯度 (注意 Padding 对应 D, H, W)
        # Pad 顺序是从最后一个维度往前数: (W_pad, H_pad, D_pad)
        # Pad: (left, right, top, bottom, front, back)

        # D方向梯度
        g_d = F.conv3d(F.pad(x_reshaped, (0, 0, 0, 0, 0, 1)), self.kernel_d)
        # H方向梯度
        g_h = F.conv3d(F.pad(x_reshaped, (0, 0, 0, 1, 0, 0)), self.kernel_h)
        # W方向梯度
        g_w = F.conv3d(F.pad(x_reshaped, (0, 1, 0, 0, 0, 0)), self.kernel_w)

        gradient = torch.abs(g_d) + torch.abs(g_h) + torch.abs(g_w)

        # 恢复形状到 [B, C, D, H, W] (计算Loss时无需转回 H,W,D，只要Target也转过来即可)
        return gradient.view(b, c, d, h, w)

    def forward(self, fused_img, source_imgs):
        """
        输入均为: [B, C, H, W, D]
        """
        # print(fused_img.shape)
        # print(source_imgs.shape)
        # 得到的梯度都是 [B, C, D, H, W] 格式
        grad_fused = self.calc_gradient_3d(fused_img)
        grads_src = [self.calc_gradient_3d(img) for img in source_imgs]

        grads_src_stack = torch.stack(grads_src, dim=1)
        target_grad, _ = torch.max(grads_src_stack, dim=1)

        return F.l1_loss(grad_fused, target_grad)


class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM3DLoss, self).__init__()
        self.window_size = window_size
        self.padding = window_size // 2

    def ssim_3d_calc(self, img1, img2):
        # 输入 img1, img2: [B, C, H, W, D]

        # 1. 维度转换 -> [B, C, D, H, W]
        img1 = img1.permute(0, 1, 4, 2, 3)
        img2 = img2.permute(0, 1, 4, 2, 3)

        # 2. 后续计算均值方差 (在 D,H,W 空间进行)
        mu1 = F.avg_pool3d(img1, self.window_size, stride=1, padding=self.padding)
        mu2 = F.avg_pool3d(img2, self.window_size, stride=1, padding=self.padding)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool3d(img1 * img1, self.window_size, stride=1, padding=self.padding) - mu1_sq
        sigma2_sq = F.avg_pool3d(img2 * img2, self.window_size, stride=1, padding=self.padding) - mu2_sq
        sigma12 = F.avg_pool3d(img1 * img2, self.window_size, stride=1, padding=self.padding) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()

    def forward(self, fused_img, source_imgs):
        loss_total = 0
        for src in source_imgs:
            loss_total += self.ssim_3d_calc(fused_img, src)
        return loss_total / len(source_imgs)


class PerceptualLoss3D(nn.Module):
    def __init__(self, device='cuda'):
        super(PerceptualLoss3D, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), vgg[x])
        self.slice2 = torch.nn.Sequential()
        for x in range(4, 9): self.slice2.add_module(str(x), vgg[x])

        self.to(device)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward_vgg(self, x_3d):
        # 输入 x_3d: [B, C, H, W, D]
        b, c, h, w, d = x_3d.shape

        # 1. 我们要把 D 混入 Batch 维度，并且要保留 C, H, W 供 VGG 使用
        # 目标: [B*D, C, H, W]

        # 先把 D 移到 B 后面: [B, D, C, H, W]
        x_permuted = x_3d.permute(0, 4, 1, 2, 3)

        # 展平前两个维度
        x_2d = x_permuted.reshape(b * d, c, h, w)

        # 如果是单通道数据 (如 CT/MRI)，复制为3通道
        if c == 1:
            x_2d = x_2d.repeat(1, 3, 1, 1)

        h1 = self.slice1(x_2d)
        h2 = self.slice2(h1)
        return h1, h2

    def forward(self, fused_img, source_imgs):
        # fused_img: [B, C, H, W, D]
        f_h1, f_h2 = self.forward_vgg(fused_img)

        src_feats = [self.forward_vgg(img) for img in source_imgs]

        loss = 0.0

        # H1 层特征最大化约束
        src_h1_stack = torch.stack([sf[0] for sf in src_feats], dim=1)
        target_h1, _ = torch.max(src_h1_stack, dim=1)
        loss += F.mse_loss(f_h1, target_h1)

        # H2 层特征最大化约束
        src_h2_stack = torch.stack([sf[1] for sf in src_feats], dim=1)
        target_h2, _ = torch.max(src_h2_stack, dim=1)
        loss += F.mse_loss(f_h2, target_h2)

        return loss


# if __name__ == '__main__':
#     device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
#     loss_grad = MaxGradientLoss3D(device)
#     loss_ssim = SSIM3DLoss()
#     loss_perceptual = PerceptualLoss3D(device)
#     img1 = torch.randn(1, 4, 128, 128, 128).to(device)
#     # img2 = torch.randn(1, 1, 128,128,128).to(device)
#     # img3 = torch.randn(1, 1, 128,128,128).to(device)
#     fused = torch.randn(1, 3, 128, 128, 128, requires_grad=True).to(device)
#     # 假设有3张图
#     imgs = [img1[:, i, :, :, :].unsqueeze(1) for i in range(img1.shape[1])]  # 都在 device 上
#     # fused = model(imgs)  # 你的融合网络
#     print(loss_grad(fused[:, 0, :, :, :].unsqueeze(1), imgs))
#     print(loss_ssim(fused[:, 1, :, :, :].unsqueeze(1), imgs))
#     print(loss_perceptual(fused[:, 2, :, :, :].unsqueeze(1), imgs))
    # # 计算总 Loss
    # L_total = 0.25 * loss_grad(fused[:, 0, :, :, :], imgs) + \
    #           0.5 * loss_ssim(fused[:, 1, :, :, :], imgs) + \
    #           0.01 * loss_perceptual(fused[:, 2, :, :, :], imgs)

    # print(L_total)
