import torch
import torch.nn as nn
conv = nn.Conv1d(1, 1, 3).cuda()
x = torch.randn(1, 1, 10).cuda()
print(conv(x).shape)
