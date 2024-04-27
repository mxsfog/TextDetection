import torch
import torch.nn as nn

__all__ = [
    "CRNN"
]


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # Combine forward pass steps
        output, _ = self.lstm(x)  # Extract only output, discard hidden state
        return self.linear(output.view(x.size(0), -1))  # Reshape and apply linear

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
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
        output = self.r(f.permute(2, 0, 1))

        return output

    def initialize_w(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
