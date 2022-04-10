from torch import nn
from torch.nn import functional as F


class Resiudal(nn.Module):
    def __init__(self, num_in, num_channels, strides=1, use1_1=False):
        super(Resiudal, self).__init__()

        mid_channels = int(num_channels // 4)
        self.conv1 = nn.Conv2d(in_channels=num_in, out_channels=mid_channels,
                               kernel_size=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=num_channels,
                               kernel_size=1)

        if use1_1:
            self.conv1_1 = nn.Conv2d(in_channels=num_in, out_channels=num_channels,
                                     kernel_size=1)
        else:
            self.conv1_1 = None

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        result_conv1 = F.relu(self.bn1(self.conv1(X)))
        result_conv2 = F.relu(self.bn2(self.conv2(result_conv1)))
        result_conv3 = self.bn3(self.conv3(result_conv2))

        if self.conv1_1:
            bypass = self.conv1_1(X)
        else:
            bypass = X

        Y = result_conv3 + bypass

        return F.relu(Y)


class Resnet_50(nn.Module):
    def __init__(self):
        super(Resnet_50, self).__init__()

        blk1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64,
                                       kernel_size=7, stride=2, padding=3),
                             nn.BatchNorm2d(64),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        def get_resnet_block(num_in, num_channels, num_residuals, is_first=False):
            blk = []

            for i in range(num_residuals):
                if not is_first and i == 0:
                    blk.append(Resiudal(num_in=num_in, num_channels=num_channels, strides=2, use1_1=True))
                elif is_first and i == 0:
                    blk.append(Resiudal(num_in=num_in, num_channels=num_channels, use1_1=True))
                else:
                    blk.append(Resiudal(num_in=num_channels, num_channels=num_channels))

            return blk

        blk2 = nn.Sequential(*get_resnet_block(num_in=64, num_channels=256, num_residuals=3, is_first=True))
        blk3 = nn.Sequential(*get_resnet_block(num_in=256, num_channels=512, num_residuals=4))
        blk4 = nn.Sequential(*get_resnet_block(num_in=512, num_channels=1024, num_residuals=6))
        blk5 = nn.Sequential(*get_resnet_block(num_in=1024, num_channels=2048, num_residuals=3))

        self.net = nn.Sequential(blk1, blk2, blk3, blk4, blk5)

    def forward(self, X):
        feature_map = self.net(X)

        return feature_map
