import torch
from torch.utils.data import DataLoader
from src.feature_extractor import FeatureExtractor, extract_patch_features
from src.memory_bank import MemoryBank
from src.dataset import MVTecDataset

DATA_ROOT = "data/mvtec/bottle"

# 初始化
extractor = FeatureExtractor().eval()
memory_bank = MemoryBank(sampling_ratio=0.1)

# 加载训练数据（只有正常图）
dataset = MVTecDataset(DATA_ROOT, split='train')
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# 提取所有正常图片的 patch 特征
all_features = []
print("提取训练集特征...")
with torch.no_grad():
    for imgs, _ in loader:
        feat1, feat2 = extractor(imgs)
        patches, _ = extract_patch_features(feat1, feat2)
        all_features.append(patches)

all_features = torch.cat(all_features, dim=0)
print(f"训练集总特征 shape: {all_features.shape}")

# 构建记忆库
memory_bank.build(all_features)

# 保存记忆库
torch.save(memory_bank.memory_bank, "memory_bank.pt")
print("✅ 记忆库已保存到 memory_bank.pt")