import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models

class Model(nn.Module):
    def __init__(self, load_model='', downsample=1, bn=True):
        super(Model, self).__init__()
        self.downsample = downsample
        self.device = torch.device('cuda:0')
        self.bn = bn
        self.features_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.features = make_layers(self.features_cfg, batch_norm=self.bn, dilation=False)
        self.avp = nn.AdaptiveAvgPool2d((1, 1))
        self.front_cfg = [512, 512]
        self.frontend = make_layers(self.front_cfg, in_channels=512, dilation=True)  # dilation = True

        #self.aspp = ASPP(in_channel=512, depth=256)

        self.backend_cfg = [256, 128]
        #self.backend = make_layers(self.backend_cfg, in_channels=256, dilation=True)
        self.backend = make_layers(self.backend_cfg, in_channels=768, dilation=True)

        #self.output_layer = nn.Conv2d(128, 1, kernel_size=1)

        self.mask_cfg = [256,256,128]
        self.maskend = make_layers(self.mask_cfg, in_channels=512, dilation=True)
       
        self.mask_layer = nn.Conv2d(128, 1, kernel_size=1)

        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.att_encoder2 = nn.Sequential(
                                          nn.Conv2d(1, 512, kernel_size=3, padding=2, dilation=2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
                                          # nn.ReLU(inplace=True),
                                          # nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
                                          )

        self.load_model = load_model
        self._init_weights()


    def forward(self, x):
        x = self.features(x)
        y = self.avp(x)
        y = y.view(y.size(0), -1)
        x = self.frontend(x)

        x1 = self.maskend(x)
        # z = self.avp(x1)
        # z = z.view(z.size(0), -1)
        x1 = self.mask_layer(x1)

        x2 = x1.sigmoid()
        x1 = self.up(x1)
        
        x_new = torch.ones_like(x2).to(self.device)
        x_new = x_new - x2
        x_new = self.att_encoder2(x_new)

        x = torch.cat((x_new,x),1)

        x = self.backend(x)

        z = self.avp(x)
        z = z.view(z.size(0), -1)

        return x, y, z, x1
    def soft_binary(self, x, k=80):
        x1 = torch.ones_like(x).to(self.device)
        x = F.relu(x - 0.0001*x1)
        x = torch.exp(-k*torch.abs(x))
        return x

    def random_init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _init_weights(self):
        if not self.load_model:
            pretrained_model = models.vgg16_bn(pretrained = True)
            self.random_init_weights()
            self.features.load_state_dict(pretrained_model.features[0:32].state_dict())
        else:
            self.load_state_dict(torch.load(self.load_model))
            print(self.load_model,' loaded!')

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)