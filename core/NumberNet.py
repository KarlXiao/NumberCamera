import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):

    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 12 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

    def forward(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, 10 * 12 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, input.size())
        input = F.grid_sample(input, grid)

        return input


class ConvModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_stride, dropout, bias=False):
        super(ConvModule, self).__init__()

        self.function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1),
            nn.Dropout2d(p=dropout, inplace=True)
        )

    def forward(self, input):
        return self.function(input)


class NumberNet(nn.Module):
    
    def __init__(self, dropout=0.2):
        super(NumberNet, self).__init__()

        self.stn = STN()

        self.conv1 = ConvModule(in_channels=3, out_channels=48, kernel_size=5, stride=1,
                                padding=2, pool_stride=2, dropout=0, bias=False)
        self.conv2 = ConvModule(in_channels=48, out_channels=64, kernel_size=5, stride=1,
                                padding=2, pool_stride=1, dropout=0, bias=False)
        self.conv3 = ConvModule(in_channels=64, out_channels=128, kernel_size=5, stride=1,
                                padding=2, pool_stride=2, dropout=0, bias=False)
        self.conv4 = ConvModule(in_channels=128, out_channels=160, kernel_size=5, stride=1,
                                padding=2, pool_stride=1, dropout=dropout, bias=False)
        self.conv5 = ConvModule(in_channels=160, out_channels=192, kernel_size=5, stride=1,
                                padding=2, pool_stride=2, dropout=0, bias=False)
        self.conv6 = ConvModule(in_channels=192, out_channels=192, kernel_size=5, stride=1,
                                padding=2, pool_stride=1, dropout=0, bias=False)
        self.conv7 = ConvModule(in_channels=192, out_channels=192, kernel_size=5, stride=1,
                                padding=2, pool_stride=2, dropout=0, bias=False)
        self.conv8 = ConvModule(in_channels=192, out_channels=192, kernel_size=5, stride=1,
                                padding=2, pool_stride=1, dropout=dropout, bias=False)

        self.fc1 = nn.Linear(in_features=3072, out_features=3072, bias=True)
        self.fc2 = nn.Linear(in_features=3072, out_features=3072, bias=True)

        self.len = nn.Linear(in_features=3072, out_features=6, bias=True)
        self.digit1 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit2 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit3 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit4 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit5 = nn.Linear(in_features=3072, out_features=11, bias=True)

        self._init_weight()

    def forward(self, input):

        input = self.stn(input)

        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        fc1 = self.fc1(torch.reshape(conv8, (conv8.shape[0], -1)))
        fc2 = self.fc2(fc1)

        l = self.len(fc2)
        digit1 = self.digit1(fc2)
        digit2 = self.digit2(fc2)
        digit3 = self.digit3(fc2)
        digit4 = self.digit4(fc2)
        digit5 = self.digit5(fc2)

        digits = torch.stack([digit1, digit2, digit3, digit4, digit5], dim=1)

        return l, digits

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
