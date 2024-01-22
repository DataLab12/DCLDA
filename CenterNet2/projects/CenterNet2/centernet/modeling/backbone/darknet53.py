import torch
from torch import nn
from detectron2.modeling.backbone import Backbone

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class Darknet53(Backbone):
    def __init__(self, block, out_features, num_classes= None):
        super(Darknet53, self).__init__()

        self.num_classes = num_classes
        self. _out_features= out_features
        self._out_feature_channels ={'res1':64, 'res2':256, 'res3':512, 'res4': 1024, 'res5': 2048}
        self._out_feature_strides= {'res1':2, 'res2':4, 'res3':8, 'res4': 16, 'res5': 32}


        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.res1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 256, stride=2)
        self.res2 = self.make_layer(block, in_channels=256, num_blocks=2)
        self.conv4 = conv_batch(256, 512, stride=2)
        self.res3 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv5 = conv_batch(512, 1024, stride=2)
        self.res4 = self.make_layer(block, in_channels=1024, num_blocks=8)
        self.conv6 = conv_batch(1024, 2048, stride=2)
        self.res5 = self.make_layer(block, in_channels=2048, num_blocks=4)

        if num_classes is not None:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):

        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}

        print(x.shape)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        out = self.res3(out)
        outputs['res3']=out
        out = self.conv5(out)
        out = self.res4(out)
        outputs['res4']=out
        out = self.conv6(out)
        out = self.res5(out)
        outputs['res5']=out

        if self.num_classes is not None:
            out = self.global_avg_pool(out)
            out = out.view(-1, 1024)
            out = self.fc(out)

        return outputs

    def make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


def darknet53(out_features, num_classes=None):
    return Darknet53(DarkResidualBlock, out_features, num_classes)


#model=darknet53(None)

#print(model)

#input=torch.rand(4, 3, 800, 800)

#print(input.shape)

#outputs=model(input)

#print(output)