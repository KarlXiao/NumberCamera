import os
import torch
import torch.nn as nn


class NumberNet(nn.Module):
    
    def __init__(self):
        super(NumberNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(160)
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(192)

        self.fc1 = nn.Linear(in_features=3072, out_features=3072, bias=True)
        self.fc2 = nn.Linear(in_features=3072, out_features=3072, bias=True)

        self.len = nn.Linear(in_features=3072, out_features=6, bias=True)
        self.digit1 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit2 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit3 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit4 = nn.Linear(in_features=3072, out_features=11, bias=True)
        self.digit5 = nn.Linear(in_features=3072, out_features=11, bias=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, input):
        conv = self.conv1(input)
        bn = self.bn1(conv)
        relu = self.max_pool(self.relu(bn))

        conv = self.conv2(relu)
        bn = self.bn2(conv)
        relu = self.relu(bn)

        conv = self.conv3(relu)
        bn = self.bn3(conv)
        relu = self.max_pool(self.relu(bn))

        conv = self.conv4(relu)
        bn = self.bn4(conv)
        relu = self.relu(bn)

        conv = self.conv5(relu)
        bn = self.bn5(conv)
        relu = self.max_pool(self.relu(bn))

        conv = self.conv6(relu)
        bn = self.bn6(conv)
        relu = self.relu(bn)

        conv = self.conv7(relu)
        bn = self.bn7(conv)
        relu = self.max_pool(self.relu(bn))

        conv = self.conv8(relu)
        bn = self.bn8(conv)
        relu = self.relu(bn)

        reshape = torch.reshape(relu, (relu.shape[0], -1))

        fc1 = self.fc1(reshape)
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
