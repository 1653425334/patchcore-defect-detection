import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练 ResNet50，我们只用它提特征，不训练它
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 取前几层，layer1 和 layer2 是我们要的中间特征
        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1  # 输出 [B, 256, 56, 56]
        )
        self.layer2 = backbone.layer2  # 输出 [B, 512, 28, 28]

        # 冻结所有参数，我们只用它提特征，不更新权重
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat1 = self.layer1(x)   # [B, 256, 56, 56]
        feat2 = self.layer2(feat1)  # [B, 512, 28, 28]
        return feat1, feat2


def get_transform():
    """图片预处理：统一尺寸 + ImageNet标准化"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet 的均值和标准差，用预训练模型必须用这个
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_patch_features(feat1, feat2):
    """
    把两层特征图融合成 patch 级别的特征向量

    思路：
    - feat1: [B, 256, 56, 56] → 先下采样到和 feat2 一样大
    - feat2: [B, 512, 28, 28]
    - 拼接后: [B, 768, 28, 28]
    - 重排成 patch 列表: [B*28*28, 768]
      即每张图有 28×28=784 个 patch，每个 patch 是 768 维向量
    """
    # 把 feat1 下采样到和 feat2 一样的空间尺寸
    feat1_resized = nn.functional.adaptive_avg_pool2d(feat1, feat2.shape[-2:])

    # 沿通道维拼接
    combined = torch.cat([feat1_resized, feat2], dim=1)  # [B, 768, 28, 28]

    B, C, H, W = combined.shape

    # 变形为 patch 列表，方便后续存入记忆库
    patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
    # [B*H*W, C] 即 [B*784, 768]

    return patches, (B, H, W)