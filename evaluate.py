import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from src.feature_extractor import FeatureExtractor, extract_patch_features
from src.memory_bank import MemoryBank
from src.dataset import MVTecDataset

DATA_ROOT = "data/mvtec/bottle"

extractor = FeatureExtractor().eval()
memory_bank = MemoryBank()
memory_bank.memory_bank = torch.load("memory_bank.pt", map_location='cpu')

dataset = MVTecDataset(DATA_ROOT, split='test')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

all_scores = []
all_labels = []

print("评估中...")
with torch.no_grad():
    for imgs, labels in loader:
        feat1, feat2 = extractor(imgs)
        patches, _ = extract_patch_features(feat1, feat2)
        scores = memory_bank.score(patches)
        anomaly_score = scores.max().item()
        all_scores.append(anomaly_score)
        all_labels.extend(labels.tolist())

auroc = roc_auc_score(all_labels, all_scores)
print(f"\n✅ Image-level AUROC: {auroc:.4f}")
print(f"（论文基准约为 0.99，CPU复现一般在 0.95+ 属于正常）")