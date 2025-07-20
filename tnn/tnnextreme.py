import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedInceptionBlock(nn.Module):
    """增强版Inception块，包含残差连接和多尺度特征融合"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0
        branch_channels = out_channels // 4
        
        # 分支1：1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels)
        )
        
        # 分支2：3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.BatchNorm2d(branch_channels))
        
        # 分支3：5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2),
            nn.BatchNorm2d(branch_channels))
        
        # 分支4：3x3最大池化
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.BatchNorm2d(branch_channels))
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        
        self.final_activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
        return self.final_activation(out + residual)

class DenseBlock(nn.Module):
    """密集连接块增强特征复用"""
    def __init__(self, in_channels, growth_rate=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, growth_rate, 3, padding=1))
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channels + growth_rate),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels + growth_rate, growth_rate, 3, padding=1))
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat([x, out1], 1))
        return torch.cat([x, out1, out2], 1)

class FrontEndNet(nn.Module):
    """增强版棋盘特征提取网络"""
    def __init__(self):
        super().__init__()
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1))
        
        # 下采样路径
        self.down1 = nn.Sequential(
            EnhancedInceptionBlock(64, 128),
            DenseBlock(128),
            nn.MaxPool2d(2))  # 20x20 -> 10x10
        
        self.down2 = nn.Sequential(
            EnhancedInceptionBlock(256, 256),  # DenseBlock输出256通道
            nn.MaxPool2d(2))  # 10x10 -> 5x5
        
        # 上采样路径
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            EnhancedInceptionBlock(256, 256))
        
        # 全局特征提取
        self.global_feat = nn.Sequential(
            EnhancedInceptionBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 1)))
        
        # 空间特征保留
        self.spatial_feat = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.init_conv(x)     # [B, 64, 20, 20]
        d1 = self.down1(x)        # [B, 256, 10, 10]
        d2 = self.down2(d1)       # [B, 256, 5, 5]
        
        u1 = self.up1(d2)         # [B, 256, 10, 10]
        
        # 跳跃连接
        skip = d1 + u1            # [B, 256, 10, 10]
        
        # 全局特征
        global_f = self.global_feat(skip)  # [B, 256, 1, 1]
        
        # 空间特征
        spatial_f = self.spatial_feat(d2)  # [B, 128, 5, 5]
        spatial_f = F.adaptive_avg_pool2d(spatial_f, (1, 1))  # [B, 128, 1, 1]
        
        # 特征融合
        combined = torch.cat([global_f, spatial_f], dim=1)  # [B, 384, 1, 1]
        return combined.view(combined.size(0), -1)  # [B, 384]

class PieceConvBlock(nn.Module):
    """专用方块处理模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.AdaptiveMaxPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.block(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        return self.fc(x)  # [B, 32]

class TNN_E(nn.Module):
    """强化版俄罗斯方块神经网络"""
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        # 特征提取网络
        self.frontend = FrontEndNet()      # 输出384维
        self.nowpiece = PieceConvBlock(4)  # 输出32维
        self.nextpiece = PieceConvBlock(4) # 输出32维
        
        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(384 + 32 + 32, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # 输出头
        self.translation_head = self._create_head(256, 10, [128, 64])  # 10个平移位置
        self.rotation_head = self._create_head(256, 4, [64])         # 4种旋转状态
        self.slide_head = self._create_head(256, 3, [64])            # 3种滑动操作

    def _create_head(self, in_features, out_features, hidden_layers):
        """创建定制化输出头"""
        layers = []
        prev_features = in_features
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_features, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(0.2))
            prev_features = hidden_size
        layers.append(nn.Linear(prev_features, out_features))
        return nn.Sequential(*layers)

    def forward(self, x_board, x_now, x_next):
        # 特征提取
        board_feat = self.frontend(x_board)  # [B, 384]
        now_feat = self.nowpiece(x_now)     # [B, 32]
        next_feat = self.nextpiece(x_next)  # [B, 32]
        
        # 特征融合
        combined = torch.cat([board_feat, now_feat, next_feat], dim=1)  # [B, 448]
        fused = self.feature_fusion(combined)  # [B, 256]
        
        # 注意力加权
        attn_weights = self.attention(fused)  # [B, 3]
        t_feat = fused * attn_weights[:, 0].unsqueeze(1)
        r_feat = fused * attn_weights[:, 1].unsqueeze(1)
        s_feat = fused * attn_weights[:, 2].unsqueeze(1)
        
        # 任务特定输出
        translation = self.translation_head(t_feat)
        rotation = self.rotation_head(r_feat)
        slide = self.slide_head(s_feat)
        
        return translation, rotation, slide
    