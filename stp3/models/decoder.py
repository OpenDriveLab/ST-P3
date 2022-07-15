import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from stp3.layers.convolutions import UpsamplingAdd, DeepLabHead


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, n_present, n_hdmap, predict_gate):
        super().__init__()
        self.perceive_hdmap = predict_gate['perceive_hdmap']
        self.predict_pedestrian = predict_gate['predict_pedestrian']
        self.predict_instance = predict_gate['predict_instance']
        self.predict_future_flow = predict_gate['predict_future_flow']
        self.planning = predict_gate['planning']

        self.n_classes = n_classes
        self.n_present = n_present
        if self.predict_instance is False and self.predict_future_flow is True:
            raise ValueError('flow cannot be True when not predicting instance')

        backbone = resnet18(pretrained=False, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
        )

        if self.predict_pedestrian:
            self.pedestrian_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, self.n_classes, kernel_size=1, padding=0),
            )

        if self.perceive_hdmap:
            self.hdmap_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2 * n_hdmap, kernel_size=1, padding=0),
            )

        if self.predict_instance:
            self.instance_offset_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )
            self.instance_center_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

        if self.planning:
            self.costvolume_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        # (H, W)
        skip_x = {'1': x}

        # (H/2, W/2)
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        skip_x['2'] = x

        # (H/4 , W/4)
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)  # (b*s, 256, 25, 25)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        segmentation_output = self.segmentation_head(x)
        pedestrian_output = self.pedestrian_head(x) if self.predict_pedestrian else None
        hdmap_output = self.hdmap_head(x.view(b, s, *x.shape[1:])[:,self.n_present-1]) if self.perceive_hdmap else None
        instance_center_output = self.instance_center_head(x) if self.predict_instance else None
        instance_offset_output = self.instance_offset_head(x) if self.predict_instance else None
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None
        costvolume = self.costvolume_head(x).squeeze(1) if self.planning else None
        return {
            'segmentation': segmentation_output.view(b, s, *segmentation_output.shape[1:]),
            'pedestrian': pedestrian_output.view(b, s, *pedestrian_output.shape[1:])
            if pedestrian_output is not None else None,
            'hdmap' : hdmap_output,
            'instance_center': instance_center_output.view(b, s, *instance_center_output.shape[1:])
            if instance_center_output is not None else None,
            'instance_offset': instance_offset_output.view(b, s, *instance_offset_output.shape[1:])
            if instance_offset_output is not None else None,
            'instance_flow': instance_future_output.view(b, s, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
            'costvolume': costvolume.view(b, s, *costvolume.shape[1:])
            if costvolume is not None else None,
        }
