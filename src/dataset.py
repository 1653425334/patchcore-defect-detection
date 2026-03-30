import os
from PIL import Image
from torch.utils.data import Dataset
from src.feature_extractor import get_transform


class MVTecDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        root_dir: bottle 文件夹路径，如 data/mvtec/bottle
        split: 'train' 只加载正常图，'test' 加载所有测试图
        """
        self.transform = get_transform()
        self.image_paths = []
        self.labels = []  # 0=正常，1=异常

        if split == 'train':
            good_dir = os.path.join(root_dir, 'train', 'good')
            for fname in os.listdir(good_dir):
                if fname.endswith(('.png', '.jpg')):
                    self.image_paths.append(os.path.join(good_dir, fname))
                    self.labels.append(0)
        else:
            test_dir = os.path.join(root_dir, 'test')
            for category in os.listdir(test_dir):
                cat_dir = os.path.join(test_dir, category)
                label = 0 if category == 'good' else 1
                for fname in os.listdir(cat_dir):
                    if fname.endswith(('.png', '.jpg')):
                        self.image_paths.append(os.path.join(cat_dir, fname))
                        self.labels.append(label)

        print(f"[{split}] 共加载 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]