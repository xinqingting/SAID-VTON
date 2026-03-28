from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# class EMASC(nn.Module):
#     """
#     改进 EMASC（简洁版，直接支持可选 pose 融合）
#     """
#     def __init__(self,
#                  in_channels: List[int],
#                  out_channels: List[int],
#                  kernel_size: int = 3,
#                  padding: int = 1,
#                  stride: int = 1,
#                  type: str = 'nonlinear',
#                  dropout: float = 0.05):
#         super().__init__()
#         self.type = type
#         self.n = len(in_channels)
#         self.conv = nn.ModuleList()
#
#         for in_ch, out_ch in zip(in_channels, out_channels):
#             if type == 'linear':
#                 layer = nn.Sequential(
#                     nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
#                     nn.BatchNorm2d(out_ch),
#                     nn.SiLU(inplace=True),
#                     nn.Dropout2d(dropout)
#                 )
#             else:  # nonlinear
#                 layer = nn.Sequential(
#                     nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
#                     nn.BatchNorm2d(in_ch),
#                     nn.SiLU(inplace=True),
#                     nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
#                     nn.BatchNorm2d(out_ch),
#                     nn.SiLU(inplace=True),
#                     nn.Dropout2d(dropout)
#                 )
#             self.conv.append(layer)
#
#         self.apply(self._init_weights)
#
#     def forward(self, x: list, pose_map: Optional[torch.Tensor] = None, fuse_layers: Optional[List[int]] = None):
#         """
#         x: list of feature maps
#         pose_map: 可选 pose 引导
#         fuse_layers: 指定哪些层融合 pose
#         """
#         out = []
#         fuse_layers = set(fuse_layers or [])
#
#         for i in range(len(x)):
#             residual = x[i]
#             y = self.conv[i](x[i])
#             if y.shape == residual.shape:
#                 y = y + 0.1 * residual  # 保留原残差
#
#             # pose 融合
#             if pose_map is not None and i in fuse_layers:
#                 pose = F.interpolate(pose_map, size=y.shape[-2:], mode='bilinear', align_corners=False)
#                 # 简单卷积降维 + 1x1 融合
#                 pose_feat = nn.Conv2d(pose.shape[1], y.shape[1], kernel_size=1, bias=True).to(y.device)(pose)
#                 # SE 注意力
#                 se = torch.sigmoid(F.adaptive_avg_pool2d(y, 1))
#                 y = y + 0.1 * (pose_feat * se)  # alpha 控制融合强度
#
#             out.append(y)
#         return out
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMASC(nn.Module):
    """
    🧠 Pose-Fusion EMASC（增强版，可训练）
    - 支持指定层融合 pose_map
    - 所有卷积参数都注册进模型，可被优化器更新
    - 保留轻量残差防止特征暗化
    - 使用 Xavier 初始化保证稳定
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_size: int = 3,
                 padding: int = 1,
                 stride: int = 1,
                 type: str = 'nonlinear',
                 dropout: float = 0.05,
                 pose_channels: int = 18,
                 fuse_layers: Optional[List[int]] = None):
        super().__init__()

        self.type = type
        self.n = len(in_channels)
        self.conv = nn.ModuleList()
        self.fuse_layers = set(fuse_layers or [])

        # pose卷积
        self.pose_convs = nn.ModuleDict({
            str(i): nn.Conv2d(pose_channels, out_channels[i], kernel_size=1, bias=True)
            for i in self.fuse_layers
        })

        # 主体卷积块
        for in_ch, out_ch in zip(in_channels, out_channels):
            if type == 'linear':
                layer = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                    nn.Dropout2d(dropout)
                )
            else:  # nonlinear
                layer = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, bias=True),
                    nn.BatchNorm2d(in_ch),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                    nn.Dropout2d(dropout)
                )
            self.conv.append(layer)

        # 初始化
        self.apply(self._init_weights)

    def forward(self, x: List[torch.Tensor], pose_map: Optional[torch.Tensor] = None):
        out = []
        for i in range(len(x)):
            residual = x[i]
            y = self.conv[i](x[i])
            if y.shape == residual.shape:
                y = y + 0.1 * residual
            if pose_map is not None and i in self.fuse_layers:
                pose = F.interpolate(pose_map, size=y.shape[-2:], mode='bilinear', align_corners=False)
                pose_feat = self.pose_convs[str(i)](pose)
                se = torch.sigmoid(F.adaptive_avg_pool2d(y, 1))
                y = y + 0.1 * (pose_feat * se)
            out.append(y)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
