import torch
from torch import nn

class DiscBlock(nn.Module):
    def __init__(self, inputs, outputs,  kernel=4, stride=2, padding=1, batchNorm=True, activation=True):
        super(DiscBlock,self).__init__()
        
        self.block1 = nn.Conv2d(inputs, outputs, kernel, stride, padding, bias=not batchNorm)
        if batchNorm: self.block2 = nn.BatchNorm2d(outputs)
        if activation: self.block3 = nn.LeakyReLU(0.2, True)

        self.batchNorm = batchNorm
        self.activation = activation


    def forward(self, x):
        out = self.block1(x)
        if self.batchNorm:
            out = self.block2(out)
        if self.activation:
            out = self.block3(out)
        # print(out.shape)
        return out
        

class Discriminator(nn.Module):
    def __init__(self, inputs=3):
        super(Discriminator,self).__init__()

        self.b1 = DiscBlock(inputs,64,batchNorm=False)
        self.b2 = DiscBlock(64,128)
        self.b3 = DiscBlock(128,256)
        self.b4 = DiscBlock(256,512,stride=1)
        self.b5 = DiscBlock(512,1,stride=1,batchNorm=False,activation=False)
                                
    def forward(self, x):
        #print(x.shape())
        y1 = self.b1(x)
        y2 = self.b2(y1)
        y3 = self.b3(y2)
        y4 = self.b4(y3)
        y5 = self.b5(y4)
        return y5