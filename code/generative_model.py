import torch
from torch import nn


class GenBlock(nn.Module):
    def __init__(self, inputs, outputs, down=True, batchNorm=True, dropout=False):
        super(GenBlock,self).__init__()

        if down:
            self.block1 = nn.Conv2d(inputs, outputs, kernel_size=4, stride=2, padding=1, bias=False)
            self.block4 = nn.LeakyReLU(0.2, True)
        else:
            self.block1 = nn.ConvTranspose2d(inputs, outputs, kernel_size=4, stride=2, padding=1, bias=False)
            self.block4 = nn.ReLU(True)
        if batchNorm:
            self.block2 = nn.BatchNorm2d(outputs)
        if dropout:
            self.block3 = nn.Dropout(0.5)

        self.batchNorm = batchNorm
        self.dropout = dropout
    
    def forward(self, x):
        out = self.block1(x)
        if self.batchNorm:
            out = self.block2(out)
        if self.dropout:
            out = self.block3(out)
        out = self.block4(out)
        return out


class Generator(nn.Module):
    def __init__(self, inputs=1):
        super(Generator,self).__init__()
        
        self.d1=  GenBlock(inputs,64,batchNorm=False)
        self.d2=  GenBlock(64,128)
        self.d3=  GenBlock(128,256)
        self.d4=  GenBlock(256,512)
        self.d5=  GenBlock(512,512)
        self.d6=  GenBlock(512,512)
        self.d7=  GenBlock(512,512)
        self.d8=  nn.Sequential(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2))
        
        
        self.u1 = GenBlock(512,512,False,dropout=True)
        self.u2 = GenBlock(1024,512,False,dropout=True)
        self.u3 = GenBlock(1024,512,False,dropout=True)
        self.u4 = GenBlock(1024,512,False)
        self.u5 = GenBlock(1024,256,False)
        self.u6 = GenBlock(512,128,False)
        self.u7 = GenBlock(256,64,False)
        self.u8 = nn.Sequential(nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        
    
    def forward(self, x):
        dd1 = self.d1(x)
        dd2 = self.d2(dd1)
        dd3 = self.d3(dd2)
        dd4 = self.d4(dd3)
        dd5 = self.d5(dd4)
        dd6 = self.d6(dd5)
        dd7 = self.d7(dd6)
        dd8 = self.d8(dd7)
        uu1 = self.u1(dd8)
        uu2 = self.u2(torch.concat([uu1,dd7],1)) #Skip Connection from dd7 to uu1
        uu3 = self.u3(torch.concat([uu2,dd6],1))
        uu4 = self.u4(torch.concat([uu3,dd5],1))
        uu5 = self.u5(torch.concat([uu4,dd4],1))
        uu6 = self.u6(torch.concat([uu5,dd3],1))
        uu7 = self.u7(torch.concat([uu6,dd2],1))
        uu8 = self.u8(torch.concat([uu7,dd1],1))
        return uu8