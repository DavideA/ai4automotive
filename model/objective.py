import torch
import torch.nn as nn

from utils import eps


class KLD(nn.Module):

    def __init__(self):
        super(KLD, self).__init__()
        self.register_buffer('eps', torch.FloatTensor([eps]))

    def forward(self, P, Q):

        # Normalize to pdf
        P /= torch.sum(P, dim=[1, 2, 3], keepdim=True) + self.eps
        Q /= torch.sum(Q, dim=[1, 2, 3], keepdim=True) + self.eps

        # DO STUFF HERE - implement KL DIVERGENCE
        kl = P * torch.log(P / (Q + self.eps) + self.eps)
        kl = torch.sum(kl, dim=[1, 2, 3])
        kl = torch.mean(kl, dim=0)
        return kl