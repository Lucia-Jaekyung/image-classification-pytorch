from typing import cast, Dict, List, Union

import torch
from torch import Tensor
from torch import nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    "VGG",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def vgg11(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg11"]))

    return model


def vgg13(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg13"]))

    return model


def vgg16(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16"]))

    return model


def vgg19(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg19"]))

    return model


def vgg11_bn(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg11"], True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg11_bn"]))

    return model


def vgg13_bn(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg13_bn"]))

    return model


def vgg16_bn(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg16_bn"]))

    return model


def vgg19_bn(pretrained=True, **kwargs) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(vgg_cfgs["vgg19"], True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["vgg19_bn"]))

    return model