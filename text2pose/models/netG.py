import torch
import torch.nn as nn


def linear(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LeakyReLU(inplace=True)
    )


def upconv4x(in_channels, out_channels, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 4, 0, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def upconv2x_hidden(in_channels, out_channels, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def upconv2x_output(in_channels, out_channels, bias=False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=bias),
        nn.Tanh()
    )


class NetG(nn.Module):
    
    def __init__(self, noise_dim, embed_dim, heatmap_channels, ngf=32):
        super(NetG, self).__init__()
        self.embed_linear = linear(embed_dim, noise_dim)
        self.combined_up1 = upconv4x(noise_dim*2, ngf*8)
        self.combined_up2 = upconv2x_hidden(ngf*8, ngf*4)
        self.combined_up3 = upconv2x_hidden(ngf*4, ngf*2)
        self.combined_up4 = upconv2x_hidden(ngf*2, ngf)
        self.combined_up5 = upconv2x_output(ngf, heatmap_channels)
    
    def forward(self, x1, x2):
        x1_noise = x1.view(x1.size(0), -1, 1, 1)
        
        x2_embed = x2.view(x2.size(0), 1, -1)
        x2_embed = self.embed_linear(x2_embed)
        x2_embed = x2_embed.view(x2_embed.size(0), -1, 1, 1)
        
        combined = torch.cat((x1_noise, x2_embed), dim=1)
        
        y = self.combined_up1(combined)
        y = self.combined_up2(y)
        y = self.combined_up3(y)
        y = self.combined_up4(y)
        y = self.combined_up5(y)
        
        return y
