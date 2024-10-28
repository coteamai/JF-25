import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from utils import plot_feature_sizes_of_efficient_net


class EfficientBackbone(nn.Module):
    def __init__(self, feature_layers, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        efficientnet = efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT,
        )
        return_layers = {
            f"{v}": f"layer_{k}" for k, v in enumerate(feature_layers)
        }
        self.features = IntermediateLayerGetter(
            efficientnet.features, return_layers=return_layers)
        self.out_channels=out_channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x, **kwargs):
        x = self.features(x)
        fpn_features = self.fpn(x)
        return fpn_features


if __name__ == "__main__":
    backbone=EfficientBackbone([4,6,8],[160,384,2560],768)
    dummy_input = torch.rand(1, 3, 512, 512)
    output = backbone(dummy_input)
    for k,v in output.items():
        print(v.shape)
    #plot_feature_sizes_of_efficient_net(efficientnet_b7(
    #        weights=EfficientNet_B7_Weights.DEFAULT,
    #))