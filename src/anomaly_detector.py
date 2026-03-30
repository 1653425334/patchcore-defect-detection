import torch
import torch.nn.functional as F
import numpy as np
import cv2
from src.feature_extractor import FeatureExtractor, extract_patch_features, get_transform
from src.memory_bank import MemoryBank
from PIL import Image


class AnomalyDetector:
    def __init__(self, memory_bank_path="memory_bank.pt"):
        self.extractor = FeatureExtractor().eval()
        self.memory_bank = MemoryBank()
        self.memory_bank.memory_bank = torch.load(memory_bank_path, map_location='cpu')
        self.transform = get_transform()
        print(f"记忆库加载完成，共 {self.memory_bank.memory_bank.shape[0]} 条特征")

    def predict(self, image: Image.Image):
        """
        输入 PIL Image，返回：
        - anomaly_score: 整张图的异常分数（越大越异常）
        - heatmap: 热力图（和原图同尺寸）
        - overlay: 热力图叠加在原图上的可视化结果
        """
        orig_w, orig_h = image.size

        # 预处理
        x = self.transform(image).unsqueeze(0)  # [1,3,224,224]

        # 提取特征
        with torch.no_grad():
            feat1, feat2 = self.extractor(x)
            patches, (B, H, W) = extract_patch_features(feat1, feat2)

        # 计算每个 patch 的异常分数
        scores = self.memory_bank.score(patches)  # [H*W]

        # 整图异常分数取最大值（最异常的patch决定整图结果）
        anomaly_score = scores.max().item()

        # 把 patch 分数 reshape 成热力图
        score_map = scores.reshape(H, W).numpy()  # [28, 28]

        # 上采样到原图尺寸
        heatmap = cv2.resize(score_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 归一化到 0-255
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

        # 转成彩色热力图（蓝→绿→红，红色=高异常）
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 叠加到原图
        orig_np = np.array(image.resize((orig_w, orig_h)))
        overlay = cv2.addWeighted(orig_np, 0.5, heatmap_color, 0.5, 0)

        return anomaly_score, heatmap_color, overlay