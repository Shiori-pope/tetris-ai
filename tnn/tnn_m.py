import torch 
import torch.nn as nn
class TNN_M(nn.Module):
    def __init__(self,prosize):
        super().__init__()
        # (1,20,10)
        self.prosize = prosize
        self.board = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),  #->(1,20,10)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5), #-> (8,4,2)
            nn.Conv2d(8, 16, kernel_size=1),  #-> (16,4,2)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),   #-> (16,2,1)
            nn.Flatten()  #-> (32)
        )
        self.bricknow = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,padding=1), #->(4,4,4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  #-> (4,1,1)
            nn.Flatten()  #-> (4)
        )
        self.bricknext = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=3,padding=1), #->(4,4,4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  #-> (4,1,1)
            nn.Flatten()  #-> (4)
        )
        self.mix = nn.Sequential(
            nn.Linear(32+4+4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        self.final = nn.Sequential(
            nn.Linear(4+prosize,24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
    def forward(self,board,now,next,pro):
        assert pro.dim() == 2 and pro.shape[1] == self.prosize, \
        f"专家特征维度不匹配，应为 [batch_size, {self.prosize}]，实际为 {list(pro.shape)}"
        board = self.board(board)
        now  =self.bricknow(now)
        next = self.bricknext(next)
        mix = torch.cat((board,now,next),dim=1)
        x = self.mix(mix)
        x = self.final(torch.cat((x,pro),dim=1))
        return x