import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from efficient_net import EfficientBackbone
from hyperparams import IN_CHANNELS,LAYERS,OUT_CHANNELS

def get_detection_model_rcnn(layers=LAYERS, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS):
    backbone = EfficientBackbone(layers, in_channels, out_channels)

    backbone.out_channels = out_channels
    feature_names = [f"layer_{i}" for i in range(len(layers))] + ['pool']
    sizes = ((32, 64, 128, 256, 512),) * len(feature_names)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(feature_names) 
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=feature_names,
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=5,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model


if __name__=="__main__":
    model = get_detection_model_rcnn().eval()
    dummy_input = torch.rand(1, 3, 512, 512)
    output = model(dummy_input)
    print(output)