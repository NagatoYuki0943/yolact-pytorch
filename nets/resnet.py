import torch
import torch.nn as nn


#-------------------------------------------------#
#   ResNet50以上使用的block
#   主干: 卷积+bn+relu -> 卷积+bn+relu -> 卷积+bn
#   短接: 卷积+bn
#   短接后有relu
#-------------------------------------------------#
class Bottleneck(nn.Module):
    # 残差结构中主分支中间和最后一层的核心数变化比例
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1    = norm_layer(planes)
        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2    = norm_layer(planes)
        self.conv3  = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3    = norm_layer(planes * 4)
        self.relu   = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 最后一层conv没有relu
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 相加后有relu
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.channels   = []
        self.inplanes   = 64
        self.norm_layer = norm_layer

        # 544, 544, 3 -> 272, 272, 64
        self.conv1      = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1        = norm_layer(64)
        self.relu       = nn.ReLU(inplace=True)
        # 272, 272, 64 -> 136, 136, 64
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers     = nn.ModuleList()
        # 136, 136, 64 -> 136, 136, 256
        self._make_layer(block, 64, layers[0])
        # 136, 136, 256 -> 68, 68, 512
        self._make_layer(block, 128, layers[1], stride=2)
        # 68, 68, 512 -> 34, 34, 1024
        self._make_layer(block, 256, layers[2], stride=2)
        # 34, 34, 1024 -> 17, 17, 2048
        self._make_layer(block, 512, layers[3], stride=2)

        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion)
            )

        layers          = [block(self.inplanes, planes, stride, downsample, self.norm_layer)]
        self.inplanes   = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)[-3:]

    def init_backbone(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
