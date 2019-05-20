import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from model.c3d import C3D


class COARSEModel(nn.Module):

    def __init__(self):
        super(COARSEModel, self).__init__()

        c3d_layers = []
        for l in list(C3D().children())[:9]:
            c3d_layers.append(l)
            if isinstance(l, nn.Conv3d):
                c3d_layers.append(nn.ReLU())
        self.c3d_layers = nn.Sequential(*c3d_layers)

        self.pool4 = nn.MaxPool3d((4, 1, 1))
        self.final_conv = nn.Conv2d(512, 1, 3, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.final_conv.weight)
        constant_(self.final_conv.bias, 0.)

    def forward(self, x):
        with torch.no_grad():
            # 1) forward through c3d layers
            h = self.c3d_layers(x)
        # 2) perform pool4
        h = self.pool4(h)
        # 3) remove temporal dimension
        h = torch.squeeze(h, dim=2)
        # 4) perform resize
        h = F.interpolate(h, scale_factor=8, mode='bilinear')
        # 5) perform final conv (with relu)
        h = self.final_conv(h)

        return h


class SingleBranchModel(nn.Module):

    def __init__(self):
        super(SingleBranchModel, self).__init__()

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(1e-3)

        self.coarse = COARSEModel()
        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            self.lrelu,
            nn.Conv2d(32, 16, 3, padding=1),
            self.lrelu,
            nn.Conv2d(16, 8, 3, padding=1),
            self.lrelu,
            nn.Conv2d(8, 1, 3, padding=1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.refine.modules():
            if isinstance(l, nn.Conv2d):
                kaiming_normal_(l.weight)
                constant_(l.bias, 0.)

    def forward(self, x_res, x_crp, X_ff):

        # Res stream
        # 1) forward through coarse
        p_res = self.coarse(x_res)
        # 2) 4x upsampling
        p_res = F.interpolate(p_res, scale_factor=4, mode='bilinear')
        # 3) Concatenate full frame (X_ff)
        p_res = torch.cat((p_res, X_ff), dim=1)
        # 4) Forward through refine layers (with relu)
        p_res = self.refine(p_res)
        p_res = self.relu(p_res)

        # Crop stream
        # 1) forward through coarse
        p_crp = self.coarse(x_crp)
        # 2) activate with relu
        p_crp = self.relu(p_crp)

        return p_crp, p_res


if __name__ == '__main__':
    fake_x_res = torch.rand(2, 3, 16, 112, 112)
    fake_x_crp = torch.rand(2, 3, 16, 112, 112)
    fake_x_ff = torch.rand(2, 3, 448, 448)
    SingleBranchModel()(fake_x_res, fake_x_crp, fake_x_ff)
