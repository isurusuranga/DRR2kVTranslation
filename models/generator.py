import torch
import torch.nn as nn
from .spectral_norm import SpectralNorm


class Generator(nn.Module):
    def __init__(self, options):
        super(Generator, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.pix_shuffle = nn.PixelShuffle(2)

        self.nf = options.nf

        self.conv1 = SpectralNorm(nn.Conv2d(in_channels=2, out_channels=self.nf, kernel_size=4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(in_channels=self.nf, out_channels=2*self.nf, kernel_size=4, stride=2,
                                            padding=1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels=2*self.nf, out_channels=4*self.nf, kernel_size=4, stride=2,
                                            padding=1))
        self.conv4 = SpectralNorm(nn.Conv2d(in_channels=4*self.nf, out_channels=8*self.nf, kernel_size=4, stride=2,
                                            padding=1))

        self.conv5 = SpectralNorm(nn.Conv2d(in_channels=8*self.nf, out_channels=16*self.nf, kernel_size=3, stride=1,
                                            padding=1))
        self.conv6 = SpectralNorm(nn.Conv2d(in_channels=16*self.nf, out_channels=8*self.nf, kernel_size=3, stride=1,
                                            padding=1))

        self.conv7 = SpectralNorm(nn.Conv2d(in_channels=8*self.nf, out_channels=4*4*self.nf, kernel_size=1, stride=1,
                                            padding=0))
        self.conv8 = SpectralNorm(nn.Conv2d(in_channels=4*self.nf, out_channels=4*2*self.nf, kernel_size=1, stride=1,
                                            padding=0))
        self.conv9 = SpectralNorm(nn.Conv2d(in_channels=2*self.nf, out_channels=4*self.nf, kernel_size=1, stride=1,
                                            padding=0))
        self.conv10 = SpectralNorm(nn.Conv2d(in_channels=self.nf, out_channels=4 * 1, kernel_size=1, stride=1,
                                             padding=0))

    def forward(self, x, gantry_angle):
        gantry_angle = gantry_angle.reshape(gantry_angle.size(0), 1, 1, 1)
        g_rep_input = gantry_angle.repeat(1, 1, x.shape[2], x.shape[3])
        updated_input_tensor = torch.cat((x, g_rep_input), 1)

        # encoder
        x = self.conv1(updated_input_tensor)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.relu(x)

        # decoder
        x = self.conv7(x)
        x = self.pix_shuffle(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.pix_shuffle(x)
        x = self.relu(x)

        x = self.conv9(x)
        x = self.pix_shuffle(x)
        x = self.relu(x)

        x = self.conv10(x)
        x = self.pix_shuffle(x)
        x = self.tanh(x)

        return x
