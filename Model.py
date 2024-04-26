import torch
import torch.nn as nn

__all__ = ["CRNN"]


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        r = self.lstm(x)
        seq_len, batch_size, input_size = r.size()
        seq_len2 = r.view(seq_len * batch_size, input_size)

        output = self.linear(seq_len2)
        output = output.view(seq_len, batch_size, -1)

        return output

class CRNN(nn.Module):
    def __int__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, (3,3), (1,1), (1,1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2), (2,2)),

            nn.Conv2d(64, 128, (3,3), (1,1), (1,1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d((2,2), (2,2)),

            nn.Conv2d(128, 256, (3,3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3,3), (1,1), (1,1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)),

            nn.Conv2d(256, 512, (3,3), (1,1), (1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3,3), (1,1), (1,1), bias=True),
            nn.ReLU(True),
            nn.MaxPool2d((2,2), (2,1), (0,1)),

            nn.Conv2d(512, 512, (2,2), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.r = nn.Sequential(
            BiLSTM(512, 256, 256),
            BiLSTM(256, 256, num_classes)
        )

        self.initialize_w()

    def forward(self, x):
        return self._forward_impl(x)

    def _forward_impl(self, x):
        f = self.conv_layers(x)
        f = f.squeeze(2)
        f = f.permute(2, 0, 1)

        output = self.r(f)

        return output

    def initialize_w(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)