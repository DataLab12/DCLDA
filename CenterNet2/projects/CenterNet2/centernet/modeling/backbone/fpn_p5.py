# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from turtle import back
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch.nn as nn

from detectron2.layers import Conv2d, ShapeSpec, get_norm

from detectron2.modeling.backbone import Backbone
from .darknet53 import darknet53
from .darknet53_csp import CSP_Darknet 
from detectron2.modeling.backbone.fpn import FPN 
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import build_resnet_backbone, BottleneckBlock

class LastLevelP6P7_P5(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.num_levels = 2
        self.in_feature = "p5"

        #print("Here i am: ", in_channels)
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):

        #print("C5 shape", c5.shape)
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]

class ExtractLocal(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, curr_args, in_channels, out_channels):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p6"
        self.maxpool= nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        self.fcn1= nn.Linear(3136, 2048)
        self.fcn2= nn.Linear(2048, 1024)
        # self.p8 =  BottleneckBlock(in_channels=in_channels, out_channels= out_channels, **curr_args)
        self.conv1=  nn.Conv2d(in_channels, 32, 3, 2, 1)
        self.conv2=  nn.Conv2d(32, out_channels, 3, 2, 1)

        for module in [self.conv1, self.conv2, self.fcn1, self.fcn2]:
            weight_init.c2_xavier_fill(module)

    def forward(self, p4):
        p8 = self.conv1(p4)
        p8 = self.conv2(p8)
        p8 = self.maxpool(p8)
        p8 = p8.reshape(p8.shape[0], -1)
        fcn_output = F.relu_(self.fcn1(p8))
        out = F.relu_(self.fcn2(fcn_output))

        return [out]

# class ExtractFeature(nn.Module):
#     """
#     This module is used in RetinaNet to generate extra layers, P6 and P7 from
#     C5 feature.
#     """

#     def __init__(self, in_feature, in_channels, out_channels):
#         super().__init__()
#         self.num_levels = 1
#         self.in_feature = in_feature
#         self.hid_channels= 2048
#         self.relu= nn.ReLU(inplace=True)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#         self.fcn1= nn.Linear(in_channels, self.hid_channels)
#         self.fcn2= nn.Linear(self.hid_channels, out_channels)

#         for module in [self.fcn1, self.fcn2]:
#             weight_init.c2_xavier_fill(module)
       

#     def forward(self, x):
#         x = self.avgpool(x)
#         print("avg pool shape: ",x.shape)
#         x= x.reshape(x.shape[0], -1)
#         print("reshape output: ",x.shape)
#         x= self.fcn1(x)
#         x = self.relu(x)
#         x= self.fcn2(x)

#         return [x]

class ExtractFeature(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_feature, in_channels, out_channels):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_feature
        self.hid_channels= 2048

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hid_channels), nn.ReLU(inplace=True),
            nn.Linear(self.hid_channels, out_channels), nn.ReLU(inplace=True))

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                 weight_init.c2_xavier_fill(m)
        # for module in self.mlp.modules():
        #     weight_init.c2_xavier_fill(module)
       

    def forward(self, x):
        x = self.avgpool(x)

        return [self.mlp(x.view(x.size(0), -1))]


@BACKBONE_REGISTRY.register()
def build_p67_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    
    curr_args ={
        'stride': 1, 
        'norm': 'FrozenBN', 
        'bottleneck_channels': 128, 
        'stride_in_1x1': True, 
        'dilation': 1, 
        'num_groups': 1}

    #print("The output from bottom up: ")
    #print(bottom_up)
    #print("Inside the bottom up: ", bottom_up["res4"])
    
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        local_block= ExtractFeature("p6", 7*7*256, 1024),
        global_block= ExtractFeature("p7", 7*7*256, 1024),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p35_resnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=None,
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p67_darknet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    bottom_up = darknet53(out_features)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone

@BACKBONE_REGISTRY.register()
def build_p67_CSP_darknet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    num_classes= cfg.MODEL.ROI_HEADS.NUM_CLASSES
    out_features  = cfg.MODEL.RESNETS.OUT_FEATURES
    bottom_up  = CSP_Darknet("/home/usr_name/Experiments/aerialAdaptation/CenterNet2/projects/CenterNet2/centernet/modeling/backbone/csp_darknet.yaml",out_features, 3, num_classes)
    print(f"Type of return : {type(bottom_up)}")
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7_P5(out_channels, out_channels),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
   
    return backbone
