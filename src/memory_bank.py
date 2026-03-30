import torch
import numpy as np
from tqdm import tqdm


class MemoryBank:
    def __init__(self, sampling_ratio=0.1):
        """
        sampling_ratio: CoreSet采样比例，保留多少比例的特征
        默认0.1即保留10%，在精度和速度间取得平衡
        """
        self.sampling_ratio = sampling_ratio
        self.memory_bank = None  # 最终存储的特征库

    def build(self, all_features: torch.Tensor):
        """
        构建记忆库
        all_features: [N, C] 所有正常图片的patch特征，N条，每条C维
        """
        print(f"原始特征数量: {all_features.shape[0]}")

        # CoreSet 采样，压缩特征库
        n_samples = max(1, int(all_features.shape[0] * self.sampling_ratio))
        print(f"采样后数量: {n_samples} (比例={self.sampling_ratio})")

        sampled = self._coreset_sample(all_features, n_samples)
        self.memory_bank = sampled
        print(f"记忆库构建完成，shape: {self.memory_bank.shape}")

    def _coreset_sample(self, features: torch.Tensor, n_samples: int):
        """
        贪心 CoreSet 采样
        每次选出距离当前已选集合最远的点，保证覆盖整个特征空间
        """
        # CPU上跑，数据量大时用float16节省内存
        features = features.float()
        n_total = features.shape[0]

        # 随机选第一个起始点
        selected_indices = [np.random.randint(0, n_total)]

        # 初始化每个点到"已选集合"的最小距离，先设为无穷大
        min_distances = torch.full((n_total,), float('inf'))

        print("CoreSet 采样中...")
        for _ in tqdm(range(n_samples - 1)):
            # 取最新加入的点
            last = features[selected_indices[-1]].unsqueeze(0)  # [1, C]

            # 计算所有点到这个点的距离
            dists = torch.cdist(features, last).squeeze(1)  # [N]

            # 更新最小距离（每个点到已选集合的最近距离）
            min_distances = torch.minimum(min_distances, dists)

            # 选出距离最大的点加入集合
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

        return features[selected_indices]

    def score(self, query_features: torch.Tensor):
        """
        计算查询特征与记忆库的异常分数
        query_features: [N, C]，一张图的所有patch特征
        返回: [N] 每个patch的异常分数（越大越异常）
        """
        assert self.memory_bank is not None, "请先调用 build() 构建记忆库"

        # 计算每个query patch到记忆库所有点的距离，取最近邻距离
        dists = torch.cdist(
            query_features.float(),
            self.memory_bank.float()
        )  # [N, M]

        # 每个patch取最近邻距离作为异常分数
        anomaly_scores, _ = dists.min(dim=1)  # [N]
        return anomaly_scores