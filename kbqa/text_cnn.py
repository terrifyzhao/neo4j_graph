import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_len, embedding_size, n_class):
        super().__init__()

        self.embedding = nn.Embedding(vocab_len, embedding_size)

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[3, embedding_size])
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[4, embedding_size])
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=[5, embedding_size])

        self.max_pool1 = nn.MaxPool1d(kernel_size=8)
        self.max_pool2 = nn.MaxPool1d(kernel_size=7)
        self.max_pool3 = nn.MaxPool1d(kernel_size=6)

        self.drop_out = nn.Dropout(0.2)
        self.full_connect = nn.Linear(300, n_class)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.unsqueeze(1)

        cnn1_out = self.cnn1(embedding).squeeze(-1)
        cnn2_out = self.cnn2(embedding).squeeze(-1)
        cnn3_out = self.cnn3(embedding).squeeze(-1)

        out1 = self.max_pool1(cnn1_out)
        out2 = self.max_pool2(cnn2_out)
        out3 = self.max_pool3(cnn3_out)

        out = torch.cat([out1, out2, out3], dim=1).squeeze(-1)

        out = self.drop_out(out)
        out = self.full_connect(out)
        # out = torch.softmax(out, dim=-1).squeeze(dim=-1)
        return out
