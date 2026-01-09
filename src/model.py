import torch
import torch.nn as nn
import torch.nn.functional as F

class KPCN(nn.Module):
    def __init__(self, input_channels=26, hidden_channels=100, output_kernel_size=21):
        super(KPCN, self).__init__()
        
        self.k = output_kernel_size
        
        layers = []
        
        # Layer 1: Input -> Hidden
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=5, padding=2))
        layers.append(nn.ReLU(inplace=True))
        
        # Layer 2 ~ 8: Hidden -> Hidden (paper)
        for _ in range(7):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            
        self.features = nn.Sequential(*layers)
        
        # Layer 9: Output Layer (Kernel Prediction)
        # Output channels = k * k (e.g., 21*21 = 441)
        self.output = nn.Conv2d(hidden_channels, self.k * self.k, kernel_size=1)
        
        # 초기화 (Xavier)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, noisy_input):
        # 1. Predict kernel weights
        feat = self.features(x)
        weights = self.output(feat) # (B, K*K, H, W)
        
        # 2. Softmax (sum of kernel weights = 1)
        weights = F.softmax(weights, dim=1)
        
        # 3. apply
        return self.apply_kernel(noisy_input, weights)

    def apply_kernel(self, img, weights):
        """
        img: (B, C, H, W) - Preprocessed Noisy Image
        weights: (B, K*K, H, W)
        """
        B, C, H, W = img.shape
        pad = self.k // 2
        
        # Padding
        img_pad = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        
        # Unfold (Im2Col)
        # patches: (B, C * K*K, H*W)
        patches = F.unfold(img_pad, kernel_size=self.k)
        patches = patches.view(B, C, self.k*self.k, H, W)
        
        # Weighted Sum
        weights = weights.unsqueeze(1) # (B, 1, K*K, H, W)
        output = (patches * weights).sum(dim=2)
        
        return output