import torch
import torch.nn.functional as F


class TextEncoder(torch.nn.Module):

    def __init__(self,
                 kernel_width,
                 channels):
        super().__init__()
        self.feature_conv = torch.nn.Conv2d(1, channels, kernel_width)
        self.weight_conv = torch.nn.Conv2d(1, channels, kernel_width)

    def forward(self, x):
        feature = self.feature_conv(x)
        weight = self.weight_conv(x)
        weight = F.softmax(weight)
        out = feature * weight
        return out


class Prado(torch.nn.Module):

    def __init__(self,
                 token_len,
                 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(token_len, 128)
        self.encoders = [TextEncoder(w, 10) for w in range(1, 6)]
        self.linear = torch.nn.Linear(100, 2)

    def forward(self, x):
        out = self.embedding(x)
        encoder_out = []
        for encoder in self.encoders:
            encoder_out.append(encoder(out))
        out = torch.cat(encoder_out)
        return self.linear(out)
