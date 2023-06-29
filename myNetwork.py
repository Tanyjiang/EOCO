import torch
import torch.nn as nn
import torch.nn.functional as F
class network(nn.Module):

    def __init__(self, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        #self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer = nn.Conv2d(128, 2, kernel_size=1)
        self.fc = nn.Linear(512, out_features=2)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.01)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

        # nn.init.xavier_uniform_(self.output_layer.weight)
        # nn.init.xavier_uniform_(self.output_layer.bias,0)
    def forward(self, input,flag=0):
        x, y, z, x1 = self.feature(input)
        x = self.output_layer(x)
        y = self.fc(y)
        #if flag == 1:
        x = F.interpolate(x, scale_factor=8)
        return x, y, z, x1


    def Incremental_learning_weight(self, numclass):
        data = self.output_layer.weight
        bias=self.output_layer.bias
        old_num=self.output_layer.out_channels
        self.output_layer=nn.Conv2d(128, out_channels=numclass+1, kernel_size=1)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            with torch.no_grad():
                nn.init.constant_(self.output_layer.bias, 0)
        with torch.no_grad():
            self.output_layer.weight[:old_num] = nn.Parameter(data)
            self.output_layer.bias[:old_num]=nn.Parameter(bias)

        weight_fc = self.fc.weight.data
        bias_fc = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        self.fc = nn.Linear(in_feature,numclass+1, bias=True)
        self.fc.weight.data[:out_feature] = weight_fc
        self.fc.bias.data[:out_feature] = bias_fc

    def Incremental_learning_head(self, numclass):
        pass


    def feature_extractor(self,inputs):
        return self.feature(inputs)
