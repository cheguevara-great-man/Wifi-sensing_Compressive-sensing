import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.nn import init
import torch.fft
fft = torch.fft










# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=7, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def UT_HAR_VGG(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model














class UT_HAR_GoogLeNet(nn.Module):
    def __init__(self, num_classes=4, aux_logits=True, init_weights=False):
        super(UT_HAR_GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x








class UT_HAR_AlexNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=False):
        super(UT_HAR_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)





def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidualV2(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidualV2, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UT_HAR_MobileNetV2(nn.Module):
    def __init__(self, num_classes=4, alpha=1.0, round_nearest=8):
        super(UT_HAR_MobileNetV2, self).__init__()
        block = InvertedResidualV2
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(1, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x







from torch import Tensor
from typing import List, Callable
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 1
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def UT_HAR_shufflenet_v2_x1_0(num_classes=7):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)
    return model










class UT_HAR_MLP(nn.Module):
    def __init__(self):
        super(UT_HAR_MLP,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(250*90,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )
        
    def forward(self,x):
        x = x.view(-1,250*90)
        x = self.fc(x)
        return x







class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
class UT_HAR_LM(nn.Module):
    def __init__(self,dim=1, attn_bias=False, proj_drop=0.):
        super(UT_HAR_LM, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)

        x = self.encoder(out)

        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out

######################时域+频域+DC=重建+分类##########################################


class LocalModulationBlock(nn.Module):
    """
    把你原来 UT_HAR_LM 里的 qkv + Spatial/ChannelOperation + dwc + proj
    封装成一个可复用模块（时域/频域都可以用）。
    """
    def __init__(self, dim, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        # 残差连接
        return out + x


class TimeDomainModule(nn.Module):
    """
    时域重建模块：卷积 + LocalModulationBlock + 卷积
    输入、输出均为时域高分辨率信号 (B, 1, T_high, 90)
    """
    def __init__(self, in_ch=1, hidden_ch=16, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.lm = LocalModulationBlock(hidden_ch, attn_bias=attn_bias, proj_drop=proj_drop)
        self.conv_out = nn.Conv2d(hidden_ch, in_ch, 3, padding=1)

    def forward(self, x):
        feat = F.relu(self.conv_in(x), inplace=True)
        feat = self.lm(feat)
        out = self.conv_out(feat)
        # 可以加一个残差，保留上采样后的基础信息
        return x + out


class FreqDomainModule(nn.Module):
    """
    频域重建模块：
    1) 对时域高分辨率信号做 FFT (沿时间维度 T)
    2) 实部/虚部 -> 2 通道张量，卷积 + LocalModulationBlock
    3) 再 iFFT 回时域，输出与时域分支同尺寸 (B, 1, T_high, 90)
    """
    def __init__(self, hidden_ch=8, attn_bias=False, proj_drop=0.):
        super().__init__()
        # 频域通道数：2 (实部+虚部)
        in_ch = 2
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.lm = LocalModulationBlock(hidden_ch, attn_bias=attn_bias, proj_drop=proj_drop)
        self.conv_out = nn.Conv2d(hidden_ch, in_ch, 3, padding=1)

    def forward(self, x_time):
        """
        x_time: (B, 1, T_high, 90)
        """
        B, C, T, N = x_time.shape
        assert C == 1, "FreqDomainModule 目前假设通道数为 1"

        # 先把 (B,1,T,N) -> (B,T,N)
        x_t = x_time.squeeze(1)  # (B, T, N)

        # 沿时间维度做 rFFT
        x_f = torch.fft.rfft(x_t, dim=1)  # (B, T_f, N), complex

        # 实部 / 虚部 -> 2 通道
        real = x_f.real.unsqueeze(1)  # (B,1,T_f,N)
        imag = x_f.imag.unsqueeze(1)  # (B,1,T_f,N)
        x_ri = torch.cat([real, imag], dim=1)  # (B,2,T_f,N)

        # 频域卷积 + LocalModulationBlock
        feat = F.relu(self.conv_in(x_ri), inplace=True)
        feat = self.lm(feat)
        feat = self.conv_out(feat)  # (B,2,T_f,N)

        # 还原为复数
        real_out, imag_out = feat.chunk(2, dim=1)
        real_out = real_out.squeeze(1)  # (B,T_f,N)
        imag_out = imag_out.squeeze(1)
        x_f_out = torch.complex(real_out, imag_out)  # (B,T_f,N)

        # iFFT 回时域 (B,T_high,N)
        x_t_out = torch.fft.irfft(x_f_out, n=T, dim=1)  # 指定 n=T 保持长度一致
        x_t_out = x_t_out.unsqueeze(1)  # (B,1,T_high,N)

        # 残差：在时域上与输入相加
        return x_time + x_t_out


class FusionModule(nn.Module):
    """
    融合模块：将时域 & 频域输出在通道维拼接，然后卷积融合为 1 通道。
    """
    def __init__(self, in_ch=2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x_time, x_freq):
        x = torch.cat([x_time, x_freq], dim=1)  # (B,2,T_high,90)
        return self.fuse(x)  # (B,1,T_high,90)

'''
class DataConsistencyLayer(nn.Module):
    """
    可指定时间维 time_dim 的数据一致性层

    x:   任意形状 [..., T, ...]，time_dim 指向时间维
    y:   与 x 在非时间维统一，最后一维是 M（采样点数）
    idx: [B, M]，B 必须是 batch 维所在维度的大小
    """
    def __init__(self, tau: float = 1.0, time_dim: int = -1, batch_dim: int = 0):
        super().__init__()
        self.tau = tau
        self.time_dim = time_dim
        self.batch_dim = batch_dim

    def forward(self, x, y, idx):
        """
        假设：
            x:  [B, L, S, T] 或更一般的，只要 time_dim 对应 T
            y:  和 x 在除时间维外的其他维度一致，最后一维为 M
            idx:[B, M]
        """
        # 先把时间维挪到最后一维，方便使用 gather/scatter_add
        x_perm = x.movedim(self.time_dim, -1)   # x_perm: [..., T]
        y_perm = y                             # 这里假设 y 已经是 [..., M]，最后一维是采样点

        # 为了和你给的 [B, L, S, T] 对齐，这里只写出典型 4 维情况：
        # 若你的 x_perm 是 [B, L, S, T]，y_perm 是 [B, L, S, M]：
        B = x_perm.size(self.batch_dim)  # 一般是 0 维

        # 推断 L,S,T
        # 这里假设 x_perm 形状为 [B, L, S, T]
        B, L, S, T = x_perm.shape
        _, _, _, M = y_perm.shape

        idx_expanded = idx[..., None, None].expand(-1, L, S, -1)  # [B, L, S, M]

        pred = x_perm.gather(-1, idx_expanded)  # [B, L, S, M]
        res  = y_perm - pred
        x_perm_updated = x_perm.scatter_add(-1, idx_expanded, self.tau * res)

        # 再把时间维挪回原来的位置
        x_updated = x_perm_updated.movedim(-1, self.time_dim)
        return x_updated

'''
class DataConsistencyLayer(nn.Module):
    """
    数据一致性层（DC 层）：
    - 假设高分辨率长度 = 低分辨率长度 * scale_factor （只在时间维度上缩放）。
    - 使用插值实现“下采样”和“上采样”，
      通过减去插值误差来提高重建与测量数据的一致性。
    """
    def __init__(self, scale_factor=2, mode="bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x_recon_hr, x_meas_lr):
        """
        x_recon_hr: (B,1,T_high,90) 重建的高分辨率信号
        x_meas_lr:  (B,1,T_low,90) 原始低分辨率测量信号
        """
        # 将高分辨率重建下采样到低分辨率
        B, C, T_low, N = x_meas_lr.shape
        x_recon_down = F.interpolate(
            x_recon_hr, size=(T_low, N),
            mode=self.mode, align_corners=False
        )

        # 计算与测量数据的误差
        error = x_recon_down - x_meas_lr  # (B,1,T_low,90)

        # 将误差上采样回高分辨率尺度
        x_error_up = F.interpolate(
            error, size=x_recon_hr.shape[2:],
            mode=self.mode, align_corners=False
        )

        # 从重建结果中减去误差
        x_dc = x_recon_hr - x_error_up
        return x_dc


class UT_HAR_Recon_Classifier(nn.Module):
    """
    完整模型：
    低分辨率信号 -> 上采样 -> 时域模块 & 频域模块 (并行) ->
    融合模块 -> DC 层得到最终重建 ->
    encoder + fc 做分类

    forward 返回：logits, x_recon_dc
    """
    def __init__(self, scale_factor=2, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.scale_factor = scale_factor

        # 上采样，只在时间维度缩放
        self.upsample = nn.Upsample(
            scale_factor=(scale_factor, 1),
            mode="bilinear",
            align_corners=False
        )

        # 时域 / 频域模块
        self.time_module = TimeDomainModule(
            in_ch=1, hidden_ch=16,
            attn_bias=attn_bias, proj_drop=proj_drop
        )
        self.freq_module = FreqDomainModule(
            hidden_ch=8,
            attn_bias=attn_bias, proj_drop=proj_drop
        )

        # 融合 + DC
        self.fusion = FusionModule(in_ch=2)
        self.dc_layer = DataConsistencyLayer(scale_factor=scale_factor, mode="bilinear")
        #self.dc_layer = DataConsistencyLayer(tau=0.5, time_dim=2, batch_dim=0)

        # ===== 下游分类部分，基本沿用你原来的 UT_HAR_LM =====
        # 注意：这里假设重建后的输入尺寸是 (B,1,250,90)，
        # 如有变化，需要根据实际尺寸调整下面 encoder 的参数。
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)   # 7 类
        )

    def forward(self, x_lr):
        """
        x_lr: 低分辨率信号 (B,1,T_low,90)
        返回:
            logits: 分类输出 (B,7)
            x_recon_dc: DC 后的重建高分辨率信号 (B,1,T_high,90)
        """
        # 1) 上采样到高分辨率
        #x_hr0 = self.upsample(x_lr)  # (B,1,T_high,90)
        x_recon = x_lr
        # 2) 时域分支 & 频域分支
        for _ in range(4):

            x_time = self.time_module(x_recon)   # (B,1,T_high,90)
            x_freq = self.freq_module(x_recon)   # (B,1,T_high,90)

        # 3) 融合 + DC
            x_fused = self.fusion(x_time, x_freq)      # (B,1,T_high,90)
            x_recon = self.dc_layer(x_fused, x_lr)  # (B,1,T_high,90)

        x_recon_dc = x_recon
        # 4) 分类（用重建后的高分辨率信号）
        feat = self.encoder(x_recon_dc)            # -> (B,96,4,4) 假定尺寸匹配
        feat = feat.view(feat.size(0), -1)         # (B,96*4*4)
        logits = self.fc(feat)                     # (B,7)

        #return logits, x_recon_dc
        return logits





class UT_HAR_Recon_Classifier_SGE(nn.Module):
    """
    完整模型：
    低分辨率信号 -> 上采样 -> 时域模块 & 频域模块 (并行) ->
    融合模块 -> DC 层得到最终重建 ->
    encoder + fc 做分类

    forward 返回：logits, x_recon_dc
    """
    def __init__(self, scale_factor=2, attn_bias=False, groups=8, proj_drop=0.):
        super().__init__()
        self.scale_factor = scale_factor


        self.upsample = nn.Upsample(
            scale_factor=(scale_factor, 1),
            mode="bilinear",
            align_corners=False
        )

        # 时域 / 频域模块
        self.time_module = TimeDomainModule(
            in_ch=1, hidden_ch=16,
            attn_bias=attn_bias, proj_drop=proj_drop
        )
        self.freq_module = FreqDomainModule(
            hidden_ch=8,
            attn_bias=attn_bias, proj_drop=proj_drop
        )

        # 融合 + DC
        self.fusion = FusionModule(in_ch=2)
        self.dc_layer = DataConsistencyLayer(scale_factor=scale_factor, mode="bilinear")
        #self.dc_layer = DataConsistencyLayer(tau=0.5, time_dim=2, batch_dim=0)


        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)   # 7 类
        )

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x_lr):
        """
        x_lr: 低分辨率信号 (B,1,T_low,90)
        返回:
            logits: 分类输出 (B,7)
            x_recon_dc: DC 后的重建高分辨率信号 (B,1,T_high,90)
        """
        # 1) 上采样到高分辨率
        #x_hr0 = self.upsample(x_lr)  # (B,1,T_high,90)
        x_recon = x_lr
        # 2) 时域分支 & 频域分支
        for _ in range(2):

            x_time = self.time_module(x_recon)   # (B,1,T_high,90)
            x_freq = self.freq_module(x_recon)   # (B,1,T_high,90)

        # 3) 融合 + DC
            x_fused = self.fusion(x_time, x_freq)      # (B,1,T_high,90)
            x_recon = self.dc_layer(x_fused, x_lr)  # (B,1,T_high,90)

        x_recon_dc = x_recon
        # 4) 分类（用重建后的高分辨率信号）
        x = self.encoder(x_recon_dc)            # -> (B,96,4,4) 假定尺寸匹配
        #####SGE####
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out



########################################################################
class UT_HAR_LM_SGE(nn.Module):
    def __init__(self,groups=8, dim=1, attn_bias=False, proj_drop=0.):
        super(UT_HAR_LM_SGE, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)

        x = self.encoder(out)
        #####SGE####
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out












class UT_HAR_SGE(nn.Module):
    def __init__(self,groups=8, dim=1, attn_bias=False, proj_drop=0.):
        super(UT_HAR_SGE, self).__init__()
        self.encoder = nn.Sequential(
            # input size: (1,250,90)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )


        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.encoder(x)
        #####SGE####
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        x = x.view(-1, 96 * 4 * 4)
        out = self.fc(x)
        return out




class UT_HAR_LeNet(nn.Module):
    def __init__(self):
        super(UT_HAR_LeNet,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (1,250,90)
            nn.Conv2d(1,32,7,stride=(3,1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,(5,4),stride=(2,2),padding=(1,0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,96,(3,3),stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(96*4*4,128),
            nn.ReLU(),
            nn.Linear(128,7)
        )
        
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,96*4*4)
        out = self.fc(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x+=identity
        x=self.relu(x)
        
        return x
    
class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    

class UT_HAR_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=7):
        super(UT_HAR_ResNet, self).__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1,3,7,stride=(3,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3,3,kernel_size=(10,11),stride=1),
            nn.ReLU()
        )
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

def UT_HAR_ResNet18():
    return UT_HAR_ResNet(Block, [2,2,2,2])
def UT_HAR_ResNet50():
    return UT_HAR_ResNet(Bottleneck, [3,4,6,3])
def UT_HAR_ResNet101():
    return UT_HAR_ResNet(Bottleneck, [3,4,23,3])


class UT_HAR_RNN(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_RNN,self).__init__()
        self.rnn = nn.RNN(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_GRU(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_GRU,self).__init__()
        self.gru = nn.GRU(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_LSTM(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_LSTM,self).__init__()
        self.lstm = nn.LSTM(90,hidden_dim,num_layers=1)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_BiLSTM(nn.Module):
    def __init__(self,hidden_dim=64):
        super(UT_HAR_BiLSTM,self).__init__()
        self.lstm = nn.LSTM(90,hidden_dim,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(hidden_dim,7)
    def forward(self,x):
        x = x.view(-1,250,90)
        x = x.permute(1,0,2)
        _, (ht,ct) = self.lstm(x)
        outputs = self.fc(ht[-1])
        return outputs


class UT_HAR_CNN_GRU(nn.Module):
    def __init__(self):
        super(UT_HAR_CNN_GRU,self).__init__()
        self.encoder = nn.Sequential(
            #input size: (250,90)
            nn.Conv1d(250,250,12,3),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,2),
            nn.ReLU(True),
            nn.Conv1d(250,250,5,1)
            # 250 x 8
        )
        self.gru = nn.GRU(8,128,num_layers=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128,7),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        # batch x 1 x 250 x 90
        x = x.view(-1,250,90)
        x = self.encoder(x)
        # batch x 250 x 8
        x = x.permute(1,0,2)
        # 250 x batch x 8
        _, ht = self.gru(x)
        outputs = self.classifier(ht[-1])
        return outputs


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 50, patch_size_h = 18, emb_size = 50*18, img_size = 250*90):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))
    
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 900, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 900,
                 drop_p = 0.,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size = 900, n_classes = 7):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))
        
class UT_HAR_ViT(nn.Sequential):
    def __init__(self,     
                in_channels = 1,
                patch_size_w = 50,
                patch_size_h = 18,
                emb_size = 900,
                img_size = 250*90,
                depth = 1,
                n_classes = 7,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
