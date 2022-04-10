from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, num_in, num_channels, strides=1, use1_1conv=True):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_in, out_channels=num_channels,
                               kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, padding=1)

        if use1_1conv:
            self.conv1_1 = nn.Conv2d(in_channels=num_in, out_channels=num_channels,
                                     kernel_size=1, stride=strides)
        else:
            self.conv1_1 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        result_conv1 = F.relu(self.bn1((self.conv1(X))))
        result_conv2 = self.bn2(self.conv2(result_conv1))

        if self.conv1_1:
            bypass = self.conv1_1(X)
        else:
            bypass = X

        Y = result_conv2 + bypass

        return F.relu(Y)


class Resnet_18(nn.Module):
    def __init__(self):
        super(Resnet_18, self).__init__()

        blk1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
                                       kernel_size=7, stride=2, padding=3),  # reduce w and h to half
                             nn.BatchNorm2d(64),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # reduce w and h to half

        def get_resnet_blk(num_in, num_channels, num_residual, is_first=False):
            blk = []
            for i in range(num_residual):
                if not is_first and i == 0:
                    blk.append(Residual(num_in=num_in, num_channels=num_channels, strides=2, use1_1conv=True))
                else:
                    blk.append(Residual(num_in=num_channels, num_channels=num_channels))

            return blk

        blk2 = nn.Sequential(*get_resnet_blk(num_in=64, num_channels=64, num_residual=2, is_first=True))
        blk3 = nn.Sequential(*get_resnet_blk(num_in=64, num_channels=128, num_residual=2))
        blk4 = nn.Sequential(*get_resnet_blk(num_in=128, num_channels=258, num_residual=2))
        blk5 = nn.Sequential(*get_resnet_blk(num_in=256, num_channels=512, num_residual=2))

        self.net = nn.Sequential(blk1, blk2, blk3, blk4, blk5)

    def forward(self, X):
        feature_map = self.net(X)

        return feature_map
