import torch
import torch.nn as nn


def linear(in_features, out_features, bias=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=bias),
        nn.LeakyReLU(inplace=True)
    )


def conv1x1(in_channels, out_channels, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
        nn.LeakyReLU(inplace=True)
    )


def downconv2x_hidden(in_channels, out_channels, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias),
        nn.LeakyReLU(inplace=True)
    )


def downconv4x_output(in_channels, out_channels, bias=False):
    return nn.Conv2d(in_channels, out_channels, 4, 4, 0, bias=bias)


class NetD(nn.Module):
    
    def __init__(self, heatmap_channels, embed_dim, ndf=32):
        super(NetD, self).__init__()
        self.heatmap_down1 = downconv2x_hidden(heatmap_channels, ndf)
        self.heatmap_down2 = downconv2x_hidden(ndf, ndf*2)
        self.heatmap_down3 = downconv2x_hidden(ndf*2, ndf*4)
        self.heatmap_down4 = downconv2x_hidden(ndf*4, ndf*8)
        self.embed_linear = linear(embed_dim, ndf*4)
        self.combined_conv = conv1x1(ndf*12, ndf*8)
        self.combined_down = downconv4x_output(ndf*8, 1)
    
    def forward(self, x1, x2):
        x1_heatmap = self.heatmap_down1(x1)
        x1_heatmap = self.heatmap_down2(x1_heatmap)
        x1_heatmap = self.heatmap_down3(x1_heatmap)
        x1_heatmap = self.heatmap_down4(x1_heatmap)
        
        x2_embed = x2.view(x2.size(0), 1, -1)
        x2_embed = self.embed_linear(x2_embed)
        x2_embed = x2_embed.view(x2_embed.size(0), -1, 1, 1)
        
        x2_tiled = torch.tile(x2_embed, (x1_heatmap.size(2), x1_heatmap.size(3)))
        
        combined = torch.cat((x1_heatmap, x2_tiled), dim=1)
        
        y = self.combined_conv(combined)
        y = self.combined_down(y)
        
        return y
