from collections import namedtuple
import Light_estimation
import torch
from torchvision import models
light_estimation=Light_estimation.Light_estimation()
light_estimation=torch.nn.DataParallel(light_estimation)
light_estimation.load_state_dict(torch.load('./checkpoint_epochcorrected_100.pth'))

class dense121(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(dense121, self).__init__()
        dense_pretrained_features = light_estimation.module.dense
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5=torch.nn.Sequential()
        self.slice6=torch.nn.Sequential()
        self.pool=light_estimation.module.pool
        color = [ light_estimation.module.color1, light_estimation.module.relu1,
             light_estimation.module.color2, light_estimation.module.relu2, light_estimation.module.color3]
        dir = [ light_estimation.module.dir1, light_estimation.module.relu3,
                 light_estimation.module.dir2, light_estimation.module.relu4, light_estimation.module.dir3]
        for x in range(6):
            self.slice1.add_module(str(x), dense_pretrained_features[x])
        for x in range(6, 8):
            self.slice2.add_module(str(x), dense_pretrained_features[x])
        for x in range(8, 10):
            self.slice3.add_module(str(x), dense_pretrained_features[x])
        for x in range(10, 12):
            self.slice4.add_module(str(x), dense_pretrained_features[x])
        for x in range(5):
            self.slice5.add_module(str(x),color[x])
        for x in range(5):
            self.slice6.add_module(str(x),dir[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h=self.pool(h).squeeze()
        Color=self.slice5(h)
        Dir=self.slice6(h)
        dense_outputs = namedtuple("denseOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3','color','dir'])
        out = dense_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3,Color,Dir)
        return out
