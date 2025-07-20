import torch 
import torch.nn as nn
class TNN_MS(nn.Module):
    def __init__(self,prosize):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Linear(prosize,64),
            nn.LeakyReLU(0.05),
            nn.Linear(64,32),
            nn.LeakyReLU(0.05),
            nn.Linear(32, 1)
        )
    def forward(self,x):
        return self.seq(x)