import torch
import torch.nn as nn
import torchvision.models as models

class Light_estimation(nn.Module):
    def __init__(self):
        super(Light_estimation, self).__init__()
        self.dense=models.densenet121(pretrained=True).features
        self.pool=nn.AvgPool2d(8)
        self.color1=nn.Linear(1024,512,bias=False)
        self.relu1=nn.ReLU()
        self.color2=nn.Linear(512,128,bias=False)
        self.relu2=nn.ReLU()
        self.color3=nn.Linear(128,5,bias=True)
        self.dir1 = nn.Linear(1024, 512, bias=False)
        self.relu3=nn.ReLU()
        self.dir2 = nn.Linear(512, 128, bias=False)
        self.relu4=nn.ReLU()
        self.dir3 = nn.Linear(128, 8, bias=True)

    def forward(self, x ):
        features=self.pool(self.dense(x))
        color=self.relu1(self.color1(features.squeeze()))
        color=self.relu2(self.color2(color))
        color=self.color3(color)
        dir=self.relu3(self.dir1(features.squeeze()))
        dir=self.relu4(self.dir2(dir))
        dir=self.dir3(dir)

        return color, dir