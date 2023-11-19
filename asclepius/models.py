import torch.nn as nn
from torchvision import models
import torch


class BreastClassifier(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(152 * 392 * in_channels, 8192)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 8)
        self.fc4 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.in_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class BreastCNN(nn.Module):
    def __init__(self, in_channels=1, image_size=(390, 150)):
        super().__init__()
        # # Size after convolution:
        # # W1XH1XD1 -> W2XH2XD2
        # # W2 = (W1 - F + 2P) / S + 1
        # # H2 = (H1 - F + 2P) / S + 1
        # # D2 = K

        # # Size after maxpool:
        # # W1XH1XD1 -> W2XH2XD2
        # # W2 = (W1 - F) / S + 1
        # # H2 = (H1 - F) / S + 1
        # # D2 = D1

        # # F = kernel size
        # # P = padding
        # # S = stride
        # # K = number of filters

        # size_after_conv1 = (
        #     (image_size[0] - 1) // 1 + 1,
        #     (image_size[1] - 1) // 1 + 1,
        #     64,
        # )
        # size_after_maxpool1 = (
        #     ((size_after_conv1[0] - 2) // (2 + 1)),
        #     ((size_after_conv1[1] - 2) // (2 + 1)),
        #     64,
        # )
        # size_after_conv2 = (
        #     (size_after_maxpool1[0] - 5) // 2,
        #     (size_after_maxpool1[1] - 5) // 2,
        #     16,
        # )
        self.image_size = image_size
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, 64, 1)  # 390x150xC -> 386x146x64
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)  # 386x146x64 -> 193x73x64
        self.conv2 = nn.Conv2d(64, 16, 5)  # 193x73x64 -> 189x69x16 -> 94x34x16
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(self._get_linear_input_size(), 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.softmax(x)
        return x

    def _get_linear_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, self.image_size[0], self.image_size[1])
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.flatten(x)
            return x.size(1)


def custom_swin_t(in_channels):
    model = models.swin_t(pretrained=True)
    model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
    model.head = nn.Linear(in_features=768, out_features=2, bias=True)
    return model


def custom_vit_b_16(in_channels):
    model = models.vit_b_16(pretrained=True)
    model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=(16, 16), stride=(16, 16))
    model.heads.head = nn.Linear(in_features=768, out_features=2, bias=True)
    return model


def custom_densenet(in_channels):
    model = models.DenseNet(12, [4, 8, 6], 64, num_classes=2)
    model.features.conv0 = nn.Conv2d(
        in_channels, 64, kernel_size=(5, 15), stride=(3, 5), padding=(3, 5), bias=False
    )

    return model
