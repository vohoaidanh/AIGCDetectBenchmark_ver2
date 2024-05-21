import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ks=1, ndim=1, norm_type=None, act_cls=None, bias=False):
        super(ConvLayer, self).__init__()
        layers = []
        if norm_type == 'Spectral':
            layers.append(nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size=ks, bias=bias)))
        else:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=ks, bias=bias))
        if act_cls is not None:
            layers.append(act_cls())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class SelfAttention(nn.Module):
    "Self-attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query = self._conv(n_channels, n_channels // 8)
        self.key = self._conv(n_channels, n_channels // 8)
        self.value = self._conv(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.0]))

    def _conv(self, n_in, n_out):
        return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type='Spectral', act_cls=None, bias=False)

    def forward(self, x):
        size = x.size()  # (batch_size, n_channels, seq_len)
        x = x.view(*size[:2], -1)  # (batch_size, n_channels, seq_len)
        f = self.query(x)  # (batch_size, n_channels // 8, seq_len)
        print('f====', f.shape)
        g = self.key(x)  # (batch_size, n_channels // 8, seq_len)
        print('g====', g.shape)

        h = self.value(x)  # (batch_size, n_channels, seq_len)
        print('h====', h.shape)

        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)  # (batch_size, seq_len, seq_len)
        o = torch.bmm(h, beta)  # (batch_size, n_channels, seq_len)
        o = self.gamma * o + x  # (batch_size, n_channels, seq_len)
        
        return o.view(*size).contiguous()  # (batch_size, n_channels, seq_len)

# Example usage
if __name__ == "__main__":
    batch_size, n_channels, seq_len = 8, 64, 10
    x = torch.randn(batch_size, n_channels, seq_len)

    self_attention = SelfAttention(n_channels)
    output = self_attention(x)

    print("Output shape:", output.shape)
# -*- coding: utf-8 -*-










