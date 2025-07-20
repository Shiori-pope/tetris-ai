import torch
import torch.nn as nn
import torch.nn.functional as F
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0
        branch_channels = out_channels // 4

        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
class FrontEndNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)  # 20×20 → 10×10

        self.incep1 = InceptionBlock(64, 128)
        self.bn1    = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,64,1, padding=0)

        self.avgpool = nn.AvgPool2d(2)  # 10×10 → 5×5

        self.conv4 = nn.Conv2d(64, 32, 1, padding=0)
        self.finalpool = nn.AvgPool2d(2)

        
        

    def forward(self, x):
        x = self.conv1(x)   # → (32, 20, 20)
        x = self.conv2(x)   # → (64, 20, 20)
        x = self.pool(x)    # → (64, 10, 10)
        x = self.incep1(x)  # → (128, 10, 10)
        x = self.bn1(F.relu(x))  # → (128, 10, 10)
        x = self.conv3(x) # → (64, 10, 10)
        x = self.avgpool(F.relu(x)) # → (64, 5, 5)
        x = self.conv4(x) # → (32, 5, 5)
        x = self.finalpool(F.relu(x)) # → (32, 2, 2)
        return x.view(x.size(0), -1)  # Flatten to (B, 128)
    
class miniNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=2),  # 输出: (8, 3, 3)
            nn.ReLU(),
            nn.Flatten(),                    # 输出: (8×3×3 = 72)
            nn.Linear(72, 32),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class miniNN33(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3,padding=1),  # 输出: (8, 4, 4)
            nn.ReLU(),
            nn.Conv2d(8,16, kernel_size=3,padding=1),  # 输出: (16, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: (16, 2, 2)
            nn.Flatten(),                   # 输出: (16*2*2=64)
            nn.Linear(64, 32),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class TetrisNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.frontend = FrontEndNet()
        self.nowpiece = miniNN()
        self.nextpiece = miniNN()
        
        # 合并特征后的共享层
        self.shared_fc = nn.Sequential(
            nn.Linear(128 + 32 + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 三个独立的输出头
        self.translation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10)  # translation: -4 到 5，编码为 0-9
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 4)   # rotation: 0-3
        )
        
        self.slide_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3)   # slide: -1,0,1，编码为 0-2
        )

    def forward(self, x_board, x_now, x_next):
        # 特征提取
        board_features = self.frontend(x_board)  # (B, 128)
        piece_features1 = self.nowpiece(x_now)  # (B, 32)
        piece_features2 = self.nextpiece(x_next)  # (B, 32)
        
        # 特征融合
        combined_features = torch.cat([board_features, piece_features1, piece_features2], dim=1)  # (B, 384)
        shared_features = self.shared_fc(combined_features)  # (B, 128)
        
        # 三个独立输出
        translation = self.translation_head(shared_features)  # (B, 10)
        rotation = self.rotation_head(shared_features)        # (B, 4)
        slide = self.slide_head(shared_features)              # (B, 3)
        
        return translation, rotation, slide
    


class TetrisNN3(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.frontend = FrontEndNet()
        self.nowpiece = miniNN33()
        self.nextpiece = miniNN33()
        
        # 合并特征后的共享层
        self.shared_fc = nn.Sequential(
            nn.Linear(128 + 32 + 32, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 三个独立的输出头
        self.translation_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10)  # translation: -4 到 5，编码为 0-9
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 4)   # rotation: 0-3
        )
        
        self.slide_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 3)   # slide: -1,0,1，编码为 0-2
        )

    def forward(self, x_board, x_now, x_next):
        # 特征提取
        board_features = self.frontend(x_board)  # (B, 128)
        piece_features1 = self.nowpiece(x_now)  # (B, 32)
        piece_features2 = self.nextpiece(x_next)  # (B, 32)
        
        # 特征融合
        combined_features = torch.cat([board_features, piece_features1, piece_features2], dim=1)  # (B, 384)
        shared_features = self.shared_fc(combined_features)  # (B, 128)
        
        # 三个独立输出
        translation = self.translation_head(shared_features)  # (B, 10)
        rotation = self.rotation_head(shared_features)        # (B, 4)
        slide = self.slide_head(shared_features)              # (B, 3)
        
        return translation, rotation, slide
    
