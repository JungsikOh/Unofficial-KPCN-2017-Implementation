import torch
from torch.utils.data import Dataset
import os
import glob

class KPCNFastDataset(Dataset):
    def __init__(self, cache_dir):
        # 해당 폴더 내의 모든 .pt 파일 검색
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {cache_dir}. Did you run generate_patches.py?")
            
        print(f"📂 FastDataset Loaded: {len(self.files)} patches from {cache_dir}")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # 파일 읽기만 수행 -> CPU 부하 낮음, 매우 빠름
        # map_location='cpu'를 명시하여 GPU 메모리 낭비 방지
        data = torch.load(self.files[idx], map_location='cpu')
        return data['input'], data['target'], data['kernel_input']