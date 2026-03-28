import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

class InversionAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 config, num_encoder_layers=2, dropout=0.5,
                 use_se=True, use_residual=True, use_outnorm=True):
        super().__init__()
        self.config = config
        self.use_se = use_se
        self.use_residual = use_residual
        self.use_outnorm = use_outnorm

        # === Transformer 编码堆叠 ===
        self.encoder_layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(num_encoder_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # === 主干 MLP ===
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # === SE 注意力模块（可选） ===
        if self.use_se:
            self.se_fc1 = nn.Linear(output_dim, output_dim // 4)
            self.se_fc2 = nn.Linear(output_dim // 4, output_dim)
            self.sigmoid = nn.Sigmoid()

        # === 残差门控模块（可选） ===
        if self.use_residual:
            self.residual_align = nn.Identity()
            self.use_residual_align = False
            self.residual_gate = nn.Parameter(torch.tensor(0.5))

        # === 输出归一化（可选） ===
        if self.use_outnorm:
            self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # 1. Transformer 编码
        for layer in self.encoder_layers:
            x = layer(x, None, None)
            x = x[0]
        x = x[:, 0, :]
        x = self.post_layernorm(x)

        # 2. MLP 映射
        y = self.layers(x)

        # 3. SE 注意力
        if self.use_se:
            se_weight = self.sigmoid(self.se_fc2(nn.GELU()(self.se_fc1(y))))
            y = y * se_weight

        # 4. 残差门控融合
        if self.use_residual:
            if y.shape[-1] != x.shape[-1]:
                if not self.use_residual_align:
                    self.residual_align = nn.Linear(x.shape[-1], y.shape[-1]).to(x.device)
                    self.use_residual_align = True
                x_aligned = self.residual_align(x)
            else:
                x_aligned = x
            y = self.residual_gate * y + (1 - self.residual_gate) * x_aligned

        # 5. 输出归一化
        if self.use_outnorm:
            y = self.output_norm(y)

        return y
# A: 原版（Baseline）
#model_A = InversionAdapter(input_dim, hidden_dim, output_dim, config, use_se=False, use_residual=False, use_outnorm=False)

# B: 加 SE
#model_B = InversionAdapter(input_dim, hidden_dim, output_dim, config,  use_se=True, use_residual=False, use_outnorm=False)

# C: 加 SE + Residual
#model_C = InversionAdapter(input_dim, hidden_dim, output_dim, config, use_se=True, use_residual=True, use_outnorm=False)

# D: Full（全部启用）
#model_D = InversionAdapter(input_dim, hidden_dim, output_dim, config, use_se=True, use_residual=True, use_outnorm=True)
