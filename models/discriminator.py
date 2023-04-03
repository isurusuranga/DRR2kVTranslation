import torch
import torch.nn as nn
from .spectral_norm import SpectralNorm


class Discriminator(nn.Module):
    """
    Discriminator
    input:
        x: one batch of data with shape of (batch_size, 1, 256, 256)
    output:
        out.squeeze: a batch of scalars indicating the predict results
    """

    def __init__(self, options):
        super(Discriminator, self).__init__()

        self.nf = options.nf

        layer1 = [SpectralNorm(nn.Conv2d(2, self.nf, 4, 2, 1)), nn.LeakyReLU(0.2)]
        curr_dim = self.nf
        self.l1 = nn.Sequential(*layer1)

        layer2 = [SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)), nn.LeakyReLU(0.2)]
        curr_dim = curr_dim * 2
        self.l2 = nn.Sequential(*layer2)

        layer3 = [SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)), nn.LeakyReLU(0.2)]
        curr_dim = curr_dim * 2
        self.l3 = nn.Sequential(*layer3)

        layer4 = [SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)), nn.LeakyReLU(0.2)]
        curr_dim = curr_dim * 2
        self.l4 = nn.Sequential(*layer4)

        layer5 = [nn.Conv2d(curr_dim, 1, 4, 1, 0)]
        self.l5 = nn.Sequential(*layer5)

    def forward(self, x, gantry_angle):
        gantry_angle = gantry_angle.reshape(gantry_angle.size(0), 1, 1, 1)
        g_rep_input = gantry_angle.repeat(1, 1, x.shape[2], x.shape[3])
        updated_input_tensor = torch.cat((x, g_rep_input), 1)

        out = self.l1(updated_input_tensor)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)

        return out.squeeze()
