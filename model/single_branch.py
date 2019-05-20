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

        self.pool4 = nn.Sequential()  # fix
        self.final_conv = nn.Sequential()  # fix

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.final_conv.weight)
        constant_(self.final_conv.bias, 0.)

    def forward(self, x):
        with torch.no_grad():
            # 1) forward through c3d layers
            # 2) perform pool4
            # 3) remove temporal dimension
            # 4) perform resize
            # 5) perform final conv (with relu)
            pass


class SingleBranchModel(nn.Module):

    def __init__(self):
        super(SingleBranchModel, self).__init__()

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(1e-3)

        self.coarse = COARSEModel()
        self.refine = nn.Sequential()

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.refine.modules():
            if isinstance(l, nn.Conv2d):
                kaiming_normal_(l.weight)
                constant_(l.bias, 0.)

    def forward(self, x_res, x_crp, X_ff):

        # Res stream
        # 1) forward through coarse
        # 2) 4x upsampling
        # 3) Concatenate full frame (X_ff)
        # 4) Forward through refine layers (with relu)

        # Crop stream
        # 1) forward through coarse
        # 2) activate with relu

        pass


if __name__ == '__main__':
    fake_x_res = torch.rand(2, 3, 16, 112, 112)
    fake_x_crp = torch.rand(2, 3, 16, 112, 112)
    fake_x_ff = torch.rand(2, 3, 448, 448)
    SingleBranchModel()(fake_x_res, fake_x_crp, fake_x_ff)
