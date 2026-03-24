#各模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1. CNN Encoder (ResNet-lite)
# =========================
class CNNEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        # x: (B*T, 1, 128, 128)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# 2. TCN Block
# =========================
class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.conv(x)
        return self.dropout(self.relu(out) + x)


class TCN(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32]
        self.layers = nn.ModuleList([
            TemporalBlock(channels, d) for d in dilations
        ])

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # → (B, C, T)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)
        return x  # (B, T, C)


# =========================
# 3. Attention + Prior
# =========================
class PriorAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.f = nn.Linear(dim, 1)

        self.g = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, S):
        # H: (B, T, C)
        # S: (B, T, 1)

        learned = torch.tanh(self.f(H))       # (B,T,1)

        # 防止先验爆炸
        S = torch.log1p(S)

        prior = self.g(S)                     # (B,T,1)

        logits = learned + self.alpha * prior
        A = torch.softmax(logits, dim=1)

        return A


# =========================
# 4. 主模型
# =========================
class MILModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CNNEncoder(32)
        self.tcn = TCN(32)
        self.attn = PriorAttention(32)

        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

    def forward(self, x, s):
        """
        x: (B, 60, 1, 128, 128)
        s: (B, 60, 1)
        """

        B, T = x.shape[:2]

        # CNN encoder
        x = x.view(B*T, 1, 128, 128)
        h = self.encoder(x)                   # (B*T, 32)
        h = h.view(B, T, -1)                  # (B, 60, 32)

        # TCN
        h = self.tcn(h)

        # Attention
        A = self.attn(h, s)                   # (B,60,1)

        # 聚合
        z = torch.sum(A * h, dim=1)           # (B,32)

        # 分类
        y = self.classifier(z)
        y = torch.sigmoid(y)

        return y, A, h

# TCN Encoder
class TCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            TemporalBlock(input_dim, hidden_dim, kernel_size=3, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            TemporalBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
        )

    def forward(self, x):
        """
        x: [B, 60, D]
        """
        x = x.transpose(1, 2)   # → [B, D, 60]
        out = self.network(x)
        out = out.transpose(1, 2)  # → [B, 60, H]
        return out
    
#Prioe-Guided MIL
import torch.nn.functional as F

class PriorTCNMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, alpha=1.0):
        super().__init__()

        self.alpha = alpha

        self.encoder = TCNEncoder(input_dim, hidden_dim)

        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, prior):
        """
        x: [B, 60, D]
        prior: [B, 60]
        """

        # -----------------------
        # TCN feature
        # -----------------------
        h = self.encoder(x)   # [B, 60, H]

        # -----------------------
        # Attention
        # -----------------------
        attn_logits = self.attn_fc(h).squeeze(-1)  # [B, 60]

        # 🔥 融合 prior（你的核心创新）
        attn_logits = attn_logits + self.alpha * prior

        attn = F.softmax(attn_logits, dim=1)

        # -----------------------
        # Bag representation
        # -----------------------
        z = torch.sum(h * attn.unsqueeze(-1), dim=1)

        # -----------------------
        # Classification
        # -----------------------
        y = torch.sigmoid(self.classifier(z)).squeeze(-1)

        return y, attn