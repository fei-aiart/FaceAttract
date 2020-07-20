import torch
import torch.nn as nn
import math

###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
class ImageAttention(nn.Module):
    def __init__(self):
        super(ImageAttention, self).__init__()
        #self.add_module('resnet',resnet(resnetBlock,in_channel=3,baseline=False))
        self.L1 = nn.Linear(in_features=1280,out_features=1280)
        self.L2 = nn.Linear(in_features=1280,out_features=1)
        self.L3 = nn.Linear(in_features=1280,out_features=1)
        #self.L4 = nn.Linear(in_features=512,out_features=1)

    def forward(self,x):
        #out = self.resnet(x)
        out = x
        out = torch.transpose(out,1,3).contiguous()
        #print out.size()
        data = out.view(-1,49,1280)#1x49x1280      
        #print 'data5',data5.size()
        out = out.view(-1,1280) #49x1280
        #print 'out1',out.size()        
        out = self.L1(out)
        out = torch.tanh(out)#49*1280
        out = self.L2(out) #49x1
        #print 'out3',out.size()        
        out = out.view(-1,49) #1x49
        #print 'out4',out.size()  
        out = nn.functional.softmax(out,dim=1)    
        out = torch.unsqueeze(out,-1)
        #4x49x1
        out = torch.sum(out*data,1) #1*1280
        #print 'out5',out.size()       
        out = self.L3(out)
        #out = nn.functional.relu(out)
        #out = self.L4(out)
        #print 'out',out.size()
        return out
        #4x5
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MV2attn(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MV2attn, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        #self.features.append(nn.AvgPool2d(input_size/32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(self.last_channel, n_class),
        # )
        self.attention = ImageAttention()


    def forward(self, x):
        #1280*7*7
        x = self.features(x)
        
        #print 'x',x.size()
        #1*1280*1*1
        #x = self.AvgPool(x)
        #x = x.view(-1, self.last_channel)
        #x = self.classifier(x)
        x= self.attention(x)
        return x

def load_weights(net, finenet):
    net.module.features._modules['0']._modules['0'].weight = finenet.module.features._modules['0']._modules['0'].weight
    net.module.features._modules['0']._modules['1'].weight = finenet.module.features._modules['0']._modules['1'].weight
    net.module.features._modules['0']._modules['1'].bias = finenet.module.features._modules['0']._modules['1'].bias
    net.module.features._modules['0']._modules['1'].running_mean = finenet.module.features._modules['0']._modules['1'].running_mean
    net.module.features._modules['0']._modules['1'].running_var = finenet.module.features._modules['0']._modules['1'].running_var
    net.module.features._modules['1'].conv._modules['0'].weight = finenet.module.features._modules['1'].conv._modules['0'].weight
    net.module.features._modules['1'].conv._modules['1'].weight = finenet.module.features._modules['1'].conv._modules['1'].weight
    net.module.features._modules['1'].conv._modules['1'].bias = finenet.module.features._modules['1'].conv._modules['1'].bias
    net.module.features._modules['1'].conv._modules['1'].running_mean = finenet.module.features._modules['1'].conv._modules['1'].running_mean
    net.module.features._modules['1'].conv._modules['1'].running_var = finenet.module.features._modules['1'].conv._modules['1'].running_var
    net.module.features._modules['1'].conv._modules['3'].weight = finenet.module.features._modules['1'].conv._modules['3'].weight
    net.module.features._modules['1'].conv._modules['4'].weight = finenet.module.features._modules['1'].conv._modules['4'].weight
    net.module.features._modules['1'].conv._modules['4'].bias = finenet.module.features._modules['1'].conv._modules['4'].bias
    net.module.features._modules['1'].conv._modules['4'].running_mean = finenet.module.features._modules['1'].conv._modules['4'].running_mean
    net.module.features._modules['1'].conv._modules['4'].running_var = finenet.module.features._modules['1'].conv._modules['4'].running_var
    net.module.features._modules['1'].conv._modules['6'].weight = finenet.module.features._modules['1'].conv._modules['6'].weight
    net.module.features._modules['1'].conv._modules['7'].weight = finenet.module.features._modules['1'].conv._modules['7'].weight
    net.module.features._modules['1'].conv._modules['7'].bias = finenet.module.features._modules['1'].conv._modules['7'].bias
    net.module.features._modules['1'].conv._modules['7'].running_mean = finenet.module.features._modules['1'].conv._modules['7'].running_mean
    net.module.features._modules['1'].conv._modules['7'].running_var = finenet.module.features._modules['1'].conv._modules['7'].running_var
    net.module.features._modules['2'].conv._modules['0'].weight = finenet.module.features._modules['2'].conv._modules['0'].weight
    net.module.features._modules['2'].conv._modules['1'].weight = finenet.module.features._modules['2'].conv._modules['1'].weight
    net.module.features._modules['2'].conv._modules['1'].bias = finenet.module.features._modules['2'].conv._modules['1'].bias
    net.module.features._modules['2'].conv._modules['1'].running_mean = finenet.module.features._modules['2'].conv._modules['1'].running_mean
    net.module.features._modules['2'].conv._modules['1'].running_var = finenet.module.features._modules['2'].conv._modules['1'].running_var
    net.module.features._modules['2'].conv._modules['3'].weight = finenet.module.features._modules['2'].conv._modules['3'].weight
    net.module.features._modules['2'].conv._modules['4'].weight = finenet.module.features._modules['2'].conv._modules['4'].weight
    net.module.features._modules['2'].conv._modules['4'].bias = finenet.module.features._modules['2'].conv._modules['4'].bias
    net.module.features._modules['2'].conv._modules['4'].running_mean = finenet.module.features._modules['2'].conv._modules['4'].running_mean
    net.module.features._modules['2'].conv._modules['4'].running_var = finenet.module.features._modules['2'].conv._modules['4'].running_var
    net.module.features._modules['2'].conv._modules['6'].weight = finenet.module.features._modules['2'].conv._modules['6'].weight
    net.module.features._modules['2'].conv._modules['7'].weight = finenet.module.features._modules['2'].conv._modules['7'].weight
    net.module.features._modules['2'].conv._modules['7'].bias = finenet.module.features._modules['2'].conv._modules['7'].bias
    net.module.features._modules['2'].conv._modules['7'].running_mean = finenet.module.features._modules['2'].conv._modules['7'].running_mean
    net.module.features._modules['2'].conv._modules['7'].running_var = finenet.module.features._modules['2'].conv._modules['7'].running_var
    net.module.features._modules['3'].conv._modules['0'].weight = finenet.module.features._modules['3'].conv._modules['0'].weight
    net.module.features._modules['3'].conv._modules['1'].weight = finenet.module.features._modules['3'].conv._modules['1'].weight
    net.module.features._modules['3'].conv._modules['1'].bias = finenet.module.features._modules['3'].conv._modules['1'].bias
    net.module.features._modules['3'].conv._modules['1'].running_mean = finenet.module.features._modules['3'].conv._modules['1'].running_mean
    net.module.features._modules['3'].conv._modules['1'].running_var = finenet.module.features._modules['3'].conv._modules['1'].running_var
    net.module.features._modules['3'].conv._modules['3'].weight = finenet.module.features._modules['3'].conv._modules['3'].weight
    net.module.features._modules['3'].conv._modules['4'].weight = finenet.module.features._modules['3'].conv._modules['4'].weight
    net.module.features._modules['3'].conv._modules['4'].bias = finenet.module.features._modules['3'].conv._modules['4'].bias
    net.module.features._modules['3'].conv._modules['4'].running_mean = finenet.module.features._modules['3'].conv._modules['4'].running_mean
    net.module.features._modules['3'].conv._modules['4'].running_var = finenet.module.features._modules['3'].conv._modules['4'].running_var
    net.module.features._modules['3'].conv._modules['6'].weight = finenet.module.features._modules['3'].conv._modules['6'].weight
    net.module.features._modules['3'].conv._modules['7'].weight = finenet.module.features._modules['3'].conv._modules['7'].weight
    net.module.features._modules['3'].conv._modules['7'].bias = finenet.module.features._modules['3'].conv._modules['7'].bias
    net.module.features._modules['3'].conv._modules['7'].running_mean = finenet.module.features._modules['3'].conv._modules['7'].running_mean
    net.module.features._modules['3'].conv._modules['7'].running_var = finenet.module.features._modules['3'].conv._modules['7'].running_var
    net.module.features._modules['4'].conv._modules['0'].weight = finenet.module.features._modules['4'].conv._modules['0'].weight
    net.module.features._modules['4'].conv._modules['1'].weight = finenet.module.features._modules['4'].conv._modules['1'].weight
    net.module.features._modules['4'].conv._modules['1'].bias = finenet.module.features._modules['4'].conv._modules['1'].bias
    net.module.features._modules['4'].conv._modules['1'].running_mean = finenet.module.features._modules['4'].conv._modules['1'].running_mean
    net.module.features._modules['4'].conv._modules['1'].running_var = finenet.module.features._modules['4'].conv._modules['1'].running_var
    net.module.features._modules['4'].conv._modules['3'].weight = finenet.module.features._modules['4'].conv._modules['3'].weight
    net.module.features._modules['4'].conv._modules['4'].weight = finenet.module.features._modules['4'].conv._modules['4'].weight
    net.module.features._modules['4'].conv._modules['4'].bias = finenet.module.features._modules['4'].conv._modules['4'].bias
    net.module.features._modules['4'].conv._modules['4'].running_mean = finenet.module.features._modules['4'].conv._modules['4'].running_mean
    net.module.features._modules['4'].conv._modules['4'].running_var = finenet.module.features._modules['4'].conv._modules['4'].running_var
    net.module.features._modules['4'].conv._modules['6'].weight = finenet.module.features._modules['4'].conv._modules['6'].weight
    net.module.features._modules['4'].conv._modules['7'].weight = finenet.module.features._modules['4'].conv._modules['7'].weight
    net.module.features._modules['4'].conv._modules['7'].bias = finenet.module.features._modules['4'].conv._modules['7'].bias
    net.module.features._modules['4'].conv._modules['7'].running_mean = finenet.module.features._modules['4'].conv._modules['7'].running_mean
    net.module.features._modules['4'].conv._modules['7'].running_var = finenet.module.features._modules['4'].conv._modules['7'].running_var
    net.module.features._modules['5'].conv._modules['0'].weight = finenet.module.features._modules['5'].conv._modules['0'].weight
    net.module.features._modules['5'].conv._modules['1'].weight = finenet.module.features._modules['5'].conv._modules['1'].weight
    net.module.features._modules['5'].conv._modules['1'].bias = finenet.module.features._modules['5'].conv._modules['1'].bias
    net.module.features._modules['5'].conv._modules['1'].running_mean = finenet.module.features._modules['5'].conv._modules['1'].running_mean
    net.module.features._modules['5'].conv._modules['1'].running_var = finenet.module.features._modules['5'].conv._modules['1'].running_var
    net.module.features._modules['5'].conv._modules['3'].weight = finenet.module.features._modules['5'].conv._modules['3'].weight
    net.module.features._modules['5'].conv._modules['4'].weight = finenet.module.features._modules['5'].conv._modules['4'].weight
    net.module.features._modules['5'].conv._modules['4'].bias = finenet.module.features._modules['5'].conv._modules['4'].bias
    net.module.features._modules['5'].conv._modules['4'].running_mean = finenet.module.features._modules['5'].conv._modules['4'].running_mean
    net.module.features._modules['5'].conv._modules['4'].running_var = finenet.module.features._modules['5'].conv._modules['4'].running_var
    net.module.features._modules['5'].conv._modules['6'].weight = finenet.module.features._modules['5'].conv._modules['6'].weight
    net.module.features._modules['5'].conv._modules['7'].weight = finenet.module.features._modules['5'].conv._modules['7'].weight
    net.module.features._modules['5'].conv._modules['7'].bias = finenet.module.features._modules['5'].conv._modules['7'].bias
    net.module.features._modules['5'].conv._modules['7'].running_mean = finenet.module.features._modules['5'].conv._modules['7'].running_mean
    net.module.features._modules['5'].conv._modules['7'].running_var = finenet.module.features._modules['5'].conv._modules['7'].running_var
    net.module.features._modules['6'].conv._modules['0'].weight = finenet.module.features._modules['6'].conv._modules['0'].weight
    net.module.features._modules['6'].conv._modules['1'].weight = finenet.module.features._modules['6'].conv._modules['1'].weight
    net.module.features._modules['6'].conv._modules['1'].bias = finenet.module.features._modules['6'].conv._modules['1'].bias
    net.module.features._modules['6'].conv._modules['1'].running_mean = finenet.module.features._modules['6'].conv._modules['1'].running_mean
    net.module.features._modules['6'].conv._modules['1'].running_var = finenet.module.features._modules['6'].conv._modules['1'].running_var
    net.module.features._modules['6'].conv._modules['3'].weight = finenet.module.features._modules['6'].conv._modules['3'].weight
    net.module.features._modules['6'].conv._modules['4'].weight = finenet.module.features._modules['6'].conv._modules['4'].weight
    net.module.features._modules['6'].conv._modules['4'].bias = finenet.module.features._modules['6'].conv._modules['4'].bias
    net.module.features._modules['6'].conv._modules['4'].running_mean = finenet.module.features._modules['6'].conv._modules['4'].running_mean
    net.module.features._modules['6'].conv._modules['4'].running_var = finenet.module.features._modules['6'].conv._modules['4'].running_var
    net.module.features._modules['6'].conv._modules['6'].weight = finenet.module.features._modules['6'].conv._modules['6'].weight
    net.module.features._modules['6'].conv._modules['7'].weight = finenet.module.features._modules['6'].conv._modules['7'].weight
    net.module.features._modules['6'].conv._modules['7'].bias = finenet.module.features._modules['6'].conv._modules['7'].bias
    net.module.features._modules['6'].conv._modules['7'].running_mean = finenet.module.features._modules['6'].conv._modules['7'].running_mean
    net.module.features._modules['6'].conv._modules['7'].running_var = finenet.module.features._modules['6'].conv._modules['7'].running_var
    net.module.features._modules['7'].conv._modules['0'].weight = finenet.module.features._modules['7'].conv._modules['0'].weight
    net.module.features._modules['7'].conv._modules['1'].weight = finenet.module.features._modules['7'].conv._modules['1'].weight
    net.module.features._modules['7'].conv._modules['1'].bias = finenet.module.features._modules['7'].conv._modules['1'].bias
    net.module.features._modules['7'].conv._modules['1'].running_mean = finenet.module.features._modules['7'].conv._modules['1'].running_mean
    net.module.features._modules['7'].conv._modules['1'].running_var = finenet.module.features._modules['7'].conv._modules['1'].running_var
    net.module.features._modules['7'].conv._modules['3'].weight = finenet.module.features._modules['7'].conv._modules['3'].weight
    net.module.features._modules['7'].conv._modules['4'].weight = finenet.module.features._modules['7'].conv._modules['4'].weight
    net.module.features._modules['7'].conv._modules['4'].bias = finenet.module.features._modules['7'].conv._modules['4'].bias
    net.module.features._modules['7'].conv._modules['4'].running_mean = finenet.module.features._modules['7'].conv._modules['4'].running_mean
    net.module.features._modules['7'].conv._modules['4'].running_var = finenet.module.features._modules['7'].conv._modules['4'].running_var
    net.module.features._modules['7'].conv._modules['6'].weight = finenet.module.features._modules['7'].conv._modules['6'].weight
    net.module.features._modules['7'].conv._modules['7'].weight = finenet.module.features._modules['7'].conv._modules['7'].weight
    net.module.features._modules['7'].conv._modules['7'].bias = finenet.module.features._modules['7'].conv._modules['7'].bias
    net.module.features._modules['7'].conv._modules['7'].running_mean = finenet.module.features._modules['7'].conv._modules['7'].running_mean
    net.module.features._modules['7'].conv._modules['7'].running_var = finenet.module.features._modules['7'].conv._modules['7'].running_var
    net.module.features._modules['8'].conv._modules['0'].weight = finenet.module.features._modules['8'].conv._modules['0'].weight
    net.module.features._modules['8'].conv._modules['1'].weight = finenet.module.features._modules['8'].conv._modules['1'].weight
    net.module.features._modules['8'].conv._modules['1'].bias = finenet.module.features._modules['8'].conv._modules['1'].bias
    net.module.features._modules['8'].conv._modules['1'].running_mean = finenet.module.features._modules['8'].conv._modules['1'].running_mean
    net.module.features._modules['8'].conv._modules['1'].running_var = finenet.module.features._modules['8'].conv._modules['1'].running_var
    net.module.features._modules['8'].conv._modules['3'].weight = finenet.module.features._modules['8'].conv._modules['3'].weight
    net.module.features._modules['8'].conv._modules['4'].weight = finenet.module.features._modules['8'].conv._modules['4'].weight
    net.module.features._modules['8'].conv._modules['4'].bias = finenet.module.features._modules['8'].conv._modules['4'].bias
    net.module.features._modules['8'].conv._modules['4'].running_mean = finenet.module.features._modules['8'].conv._modules['4'].running_mean
    net.module.features._modules['8'].conv._modules['4'].running_var = finenet.module.features._modules['8'].conv._modules['4'].running_var
    net.module.features._modules['8'].conv._modules['6'].weight = finenet.module.features._modules['8'].conv._modules['6'].weight
    net.module.features._modules['8'].conv._modules['7'].weight = finenet.module.features._modules['8'].conv._modules['7'].weight
    net.module.features._modules['8'].conv._modules['7'].bias = finenet.module.features._modules['8'].conv._modules['7'].bias
    net.module.features._modules['8'].conv._modules['7'].running_mean = finenet.module.features._modules['8'].conv._modules['7'].running_mean
    net.module.features._modules['8'].conv._modules['7'].running_var = finenet.module.features._modules['8'].conv._modules['7'].running_var
    net.module.features._modules['9'].conv._modules['0'].weight = finenet.module.features._modules['9'].conv._modules['0'].weight
    net.module.features._modules['9'].conv._modules['1'].weight = finenet.module.features._modules['9'].conv._modules['1'].weight
    net.module.features._modules['9'].conv._modules['1'].bias = finenet.module.features._modules['9'].conv._modules['1'].bias
    net.module.features._modules['9'].conv._modules['1'].running_mean = finenet.module.features._modules['9'].conv._modules['1'].running_mean
    net.module.features._modules['9'].conv._modules['1'].running_var = finenet.module.features._modules['9'].conv._modules['1'].running_var
    net.module.features._modules['9'].conv._modules['3'].weight = finenet.module.features._modules['9'].conv._modules['3'].weight
    net.module.features._modules['9'].conv._modules['4'].weight = finenet.module.features._modules['9'].conv._modules['4'].weight
    net.module.features._modules['9'].conv._modules['4'].bias = finenet.module.features._modules['9'].conv._modules['4'].bias
    net.module.features._modules['9'].conv._modules['4'].running_mean = finenet.module.features._modules['9'].conv._modules['4'].running_mean
    net.module.features._modules['9'].conv._modules['4'].running_var = finenet.module.features._modules['9'].conv._modules['4'].running_var
    net.module.features._modules['9'].conv._modules['6'].weight = finenet.module.features._modules['9'].conv._modules['6'].weight
    net.module.features._modules['9'].conv._modules['7'].weight = finenet.module.features._modules['9'].conv._modules['7'].weight
    net.module.features._modules['9'].conv._modules['7'].bias = finenet.module.features._modules['9'].conv._modules['7'].bias
    net.module.features._modules['9'].conv._modules['7'].running_mean = finenet.module.features._modules['9'].conv._modules['7'].running_mean
    net.module.features._modules['9'].conv._modules['7'].running_var = finenet.module.features._modules['9'].conv._modules['7'].running_var
    net.module.features._modules['10'].conv._modules['0'].weight = finenet.module.features._modules['10'].conv._modules['0'].weight
    net.module.features._modules['10'].conv._modules['1'].weight = finenet.module.features._modules['10'].conv._modules['1'].weight
    net.module.features._modules['10'].conv._modules['1'].bias = finenet.module.features._modules['10'].conv._modules['1'].bias
    net.module.features._modules['10'].conv._modules['1'].running_mean = finenet.module.features._modules['10'].conv._modules['1'].running_mean
    net.module.features._modules['10'].conv._modules['1'].running_var = finenet.module.features._modules['10'].conv._modules['1'].running_var
    net.module.features._modules['10'].conv._modules['3'].weight = finenet.module.features._modules['10'].conv._modules['3'].weight
    net.module.features._modules['10'].conv._modules['4'].weight = finenet.module.features._modules['10'].conv._modules['4'].weight
    net.module.features._modules['10'].conv._modules['4'].bias = finenet.module.features._modules['10'].conv._modules['4'].bias
    net.module.features._modules['10'].conv._modules['4'].running_mean = finenet.module.features._modules['10'].conv._modules['4'].running_mean
    net.module.features._modules['10'].conv._modules['4'].running_var = finenet.module.features._modules['10'].conv._modules['4'].running_var
    net.module.features._modules['10'].conv._modules['6'].weight = finenet.module.features._modules['10'].conv._modules['6'].weight
    net.module.features._modules['10'].conv._modules['7'].weight = finenet.module.features._modules['10'].conv._modules['7'].weight
    net.module.features._modules['10'].conv._modules['7'].bias = finenet.module.features._modules['10'].conv._modules['7'].bias
    net.module.features._modules['10'].conv._modules['7'].running_mean = finenet.module.features._modules['10'].conv._modules['7'].running_mean
    net.module.features._modules['10'].conv._modules['7'].running_var = finenet.module.features._modules['10'].conv._modules['7'].running_var
    net.module.features._modules['11'].conv._modules['0'].weight = finenet.module.features._modules['11'].conv._modules['0'].weight
    net.module.features._modules['11'].conv._modules['1'].weight = finenet.module.features._modules['11'].conv._modules['1'].weight
    net.module.features._modules['11'].conv._modules['1'].bias = finenet.module.features._modules['11'].conv._modules['1'].bias
    net.module.features._modules['11'].conv._modules['1'].running_mean = finenet.module.features._modules['11'].conv._modules['1'].running_mean
    net.module.features._modules['11'].conv._modules['1'].running_var = finenet.module.features._modules['11'].conv._modules['1'].running_var
    net.module.features._modules['11'].conv._modules['3'].weight = finenet.module.features._modules['11'].conv._modules['3'].weight
    net.module.features._modules['11'].conv._modules['4'].weight = finenet.module.features._modules['11'].conv._modules['4'].weight
    net.module.features._modules['11'].conv._modules['4'].bias = finenet.module.features._modules['11'].conv._modules['4'].bias
    net.module.features._modules['11'].conv._modules['4'].running_mean = finenet.module.features._modules['11'].conv._modules['4'].running_mean
    net.module.features._modules['11'].conv._modules['4'].running_var = finenet.module.features._modules['11'].conv._modules['4'].running_var
    net.module.features._modules['11'].conv._modules['6'].weight = finenet.module.features._modules['11'].conv._modules['6'].weight
    net.module.features._modules['11'].conv._modules['7'].weight = finenet.module.features._modules['11'].conv._modules['7'].weight
    net.module.features._modules['11'].conv._modules['7'].bias = finenet.module.features._modules['11'].conv._modules['7'].bias
    net.module.features._modules['11'].conv._modules['7'].running_mean = finenet.module.features._modules['11'].conv._modules['7'].running_mean
    net.module.features._modules['11'].conv._modules['7'].running_var = finenet.module.features._modules['11'].conv._modules['7'].running_var
    net.module.features._modules['12'].conv._modules['0'].weight = finenet.module.features._modules['12'].conv._modules['0'].weight
    net.module.features._modules['12'].conv._modules['1'].weight = finenet.module.features._modules['12'].conv._modules['1'].weight
    net.module.features._modules['12'].conv._modules['1'].bias = finenet.module.features._modules['12'].conv._modules['1'].bias
    net.module.features._modules['12'].conv._modules['1'].running_mean = finenet.module.features._modules['12'].conv._modules['1'].running_mean
    net.module.features._modules['12'].conv._modules['1'].running_var = finenet.module.features._modules['12'].conv._modules['1'].running_var
    net.module.features._modules['12'].conv._modules['3'].weight = finenet.module.features._modules['12'].conv._modules['3'].weight
    net.module.features._modules['12'].conv._modules['4'].weight = finenet.module.features._modules['12'].conv._modules['4'].weight
    net.module.features._modules['12'].conv._modules['4'].bias = finenet.module.features._modules['12'].conv._modules['4'].bias
    net.module.features._modules['12'].conv._modules['4'].running_mean = finenet.module.features._modules['12'].conv._modules['4'].running_mean
    net.module.features._modules['12'].conv._modules['4'].running_var = finenet.module.features._modules['12'].conv._modules['4'].running_var
    net.module.features._modules['12'].conv._modules['6'].weight = finenet.module.features._modules['12'].conv._modules['6'].weight
    net.module.features._modules['12'].conv._modules['7'].weight = finenet.module.features._modules['12'].conv._modules['7'].weight
    net.module.features._modules['12'].conv._modules['7'].bias = finenet.module.features._modules['12'].conv._modules['7'].bias
    net.module.features._modules['12'].conv._modules['7'].running_mean = finenet.module.features._modules['12'].conv._modules['7'].running_mean
    net.module.features._modules['12'].conv._modules['7'].running_var = finenet.module.features._modules['12'].conv._modules['7'].running_var
    net.module.features._modules['13'].conv._modules['0'].weight = finenet.module.features._modules['13'].conv._modules['0'].weight
    net.module.features._modules['13'].conv._modules['1'].weight = finenet.module.features._modules['13'].conv._modules['1'].weight
    net.module.features._modules['13'].conv._modules['1'].bias = finenet.module.features._modules['13'].conv._modules['1'].bias
    net.module.features._modules['13'].conv._modules['1'].running_mean = finenet.module.features._modules['13'].conv._modules['1'].running_mean
    net.module.features._modules['13'].conv._modules['1'].running_var = finenet.module.features._modules['13'].conv._modules['1'].running_var
    net.module.features._modules['13'].conv._modules['3'].weight = finenet.module.features._modules['13'].conv._modules['3'].weight
    net.module.features._modules['13'].conv._modules['4'].weight = finenet.module.features._modules['13'].conv._modules['4'].weight
    net.module.features._modules['13'].conv._modules['4'].bias = finenet.module.features._modules['13'].conv._modules['4'].bias
    net.module.features._modules['13'].conv._modules['4'].running_mean = finenet.module.features._modules['13'].conv._modules['4'].running_mean
    net.module.features._modules['13'].conv._modules['4'].running_var = finenet.module.features._modules['13'].conv._modules['4'].running_var
    net.module.features._modules['13'].conv._modules['6'].weight = finenet.module.features._modules['13'].conv._modules['6'].weight
    net.module.features._modules['13'].conv._modules['7'].weight = finenet.module.features._modules['13'].conv._modules['7'].weight
    net.module.features._modules['13'].conv._modules['7'].bias = finenet.module.features._modules['13'].conv._modules['7'].bias
    net.module.features._modules['13'].conv._modules['7'].running_mean = finenet.module.features._modules['13'].conv._modules['7'].running_mean
    net.module.features._modules['13'].conv._modules['7'].running_var = finenet.module.features._modules['13'].conv._modules['7'].running_var
    net.module.features._modules['14'].conv._modules['0'].weight = finenet.module.features._modules['14'].conv._modules['0'].weight
    net.module.features._modules['14'].conv._modules['1'].weight = finenet.module.features._modules['14'].conv._modules['1'].weight
    net.module.features._modules['14'].conv._modules['1'].bias = finenet.module.features._modules['14'].conv._modules['1'].bias
    net.module.features._modules['14'].conv._modules['1'].running_mean = finenet.module.features._modules['14'].conv._modules['1'].running_mean
    net.module.features._modules['14'].conv._modules['1'].running_var = finenet.module.features._modules['14'].conv._modules['1'].running_var
    net.module.features._modules['14'].conv._modules['3'].weight = finenet.module.features._modules['14'].conv._modules['3'].weight
    net.module.features._modules['14'].conv._modules['4'].weight = finenet.module.features._modules['14'].conv._modules['4'].weight
    net.module.features._modules['14'].conv._modules['4'].bias = finenet.module.features._modules['14'].conv._modules['4'].bias
    net.module.features._modules['14'].conv._modules['4'].running_mean = finenet.module.features._modules['14'].conv._modules['4'].running_mean
    net.module.features._modules['14'].conv._modules['4'].running_var = finenet.module.features._modules['14'].conv._modules['4'].running_var
    net.module.features._modules['14'].conv._modules['6'].weight = finenet.module.features._modules['14'].conv._modules['6'].weight
    net.module.features._modules['14'].conv._modules['7'].weight = finenet.module.features._modules['14'].conv._modules['7'].weight
    net.module.features._modules['14'].conv._modules['7'].bias = finenet.module.features._modules['14'].conv._modules['7'].bias
    net.module.features._modules['14'].conv._modules['7'].running_mean = finenet.module.features._modules['14'].conv._modules['7'].running_mean
    net.module.features._modules['14'].conv._modules['7'].running_var = finenet.module.features._modules['14'].conv._modules['7'].running_var
    net.module.features._modules['15'].conv._modules['0'].weight = finenet.module.features._modules['15'].conv._modules['0'].weight
    net.module.features._modules['15'].conv._modules['1'].weight = finenet.module.features._modules['15'].conv._modules['1'].weight
    net.module.features._modules['15'].conv._modules['1'].bias = finenet.module.features._modules['15'].conv._modules['1'].bias
    net.module.features._modules['15'].conv._modules['1'].running_mean = finenet.module.features._modules['15'].conv._modules['1'].running_mean
    net.module.features._modules['15'].conv._modules['1'].running_var = finenet.module.features._modules['15'].conv._modules['1'].running_var
    net.module.features._modules['15'].conv._modules['3'].weight = finenet.module.features._modules['15'].conv._modules['3'].weight
    net.module.features._modules['15'].conv._modules['4'].weight = finenet.module.features._modules['15'].conv._modules['4'].weight
    net.module.features._modules['15'].conv._modules['4'].bias = finenet.module.features._modules['15'].conv._modules['4'].bias
    net.module.features._modules['15'].conv._modules['4'].running_mean = finenet.module.features._modules['15'].conv._modules['4'].running_mean
    net.module.features._modules['15'].conv._modules['4'].running_var = finenet.module.features._modules['15'].conv._modules['4'].running_var
    net.module.features._modules['15'].conv._modules['6'].weight = finenet.module.features._modules['15'].conv._modules['6'].weight
    net.module.features._modules['15'].conv._modules['7'].weight = finenet.module.features._modules['15'].conv._modules['7'].weight
    net.module.features._modules['15'].conv._modules['7'].bias = finenet.module.features._modules['15'].conv._modules['7'].bias
    net.module.features._modules['15'].conv._modules['7'].running_mean = finenet.module.features._modules['15'].conv._modules['7'].running_mean
    net.module.features._modules['15'].conv._modules['7'].running_var = finenet.module.features._modules['15'].conv._modules['7'].running_var
    net.module.features._modules['16'].conv._modules['0'].weight = finenet.module.features._modules['16'].conv._modules['0'].weight
    net.module.features._modules['16'].conv._modules['1'].weight = finenet.module.features._modules['16'].conv._modules['1'].weight
    net.module.features._modules['16'].conv._modules['1'].bias = finenet.module.features._modules['16'].conv._modules['1'].bias
    net.module.features._modules['16'].conv._modules['1'].running_mean = finenet.module.features._modules['16'].conv._modules['1'].running_mean
    net.module.features._modules['16'].conv._modules['1'].running_var = finenet.module.features._modules['16'].conv._modules['1'].running_var
    net.module.features._modules['16'].conv._modules['3'].weight = finenet.module.features._modules['16'].conv._modules['3'].weight
    net.module.features._modules['16'].conv._modules['4'].weight = finenet.module.features._modules['16'].conv._modules['4'].weight
    net.module.features._modules['16'].conv._modules['4'].bias = finenet.module.features._modules['16'].conv._modules['4'].bias
    net.module.features._modules['16'].conv._modules['4'].running_mean = finenet.module.features._modules['16'].conv._modules['4'].running_mean
    net.module.features._modules['16'].conv._modules['4'].running_var = finenet.module.features._modules['16'].conv._modules['4'].running_var
    net.module.features._modules['16'].conv._modules['6'].weight = finenet.module.features._modules['16'].conv._modules['6'].weight
    net.module.features._modules['16'].conv._modules['7'].weight = finenet.module.features._modules['16'].conv._modules['7'].weight
    net.module.features._modules['16'].conv._modules['7'].bias = finenet.module.features._modules['16'].conv._modules['7'].bias
    net.module.features._modules['16'].conv._modules['7'].running_mean = finenet.module.features._modules['16'].conv._modules['7'].running_mean
    net.module.features._modules['16'].conv._modules['7'].running_var = finenet.module.features._modules['16'].conv._modules['7'].running_var
    net.module.features._modules['17'].conv._modules['0'].weight = finenet.module.features._modules['17'].conv._modules['0'].weight
    net.module.features._modules['17'].conv._modules['1'].weight = finenet.module.features._modules['17'].conv._modules['1'].weight
    net.module.features._modules['17'].conv._modules['1'].bias = finenet.module.features._modules['17'].conv._modules['1'].bias
    net.module.features._modules['17'].conv._modules['1'].running_mean = finenet.module.features._modules['17'].conv._modules['1'].running_mean
    net.module.features._modules['17'].conv._modules['1'].running_var = finenet.module.features._modules['17'].conv._modules['1'].running_var
    net.module.features._modules['17'].conv._modules['3'].weight = finenet.module.features._modules['17'].conv._modules['3'].weight
    net.module.features._modules['17'].conv._modules['4'].weight = finenet.module.features._modules['17'].conv._modules['4'].weight
    net.module.features._modules['17'].conv._modules['4'].bias = finenet.module.features._modules['17'].conv._modules['4'].bias
    net.module.features._modules['17'].conv._modules['4'].running_mean = finenet.module.features._modules['17'].conv._modules['4'].running_mean
    net.module.features._modules['17'].conv._modules['4'].running_var = finenet.module.features._modules['17'].conv._modules['4'].running_var
    net.module.features._modules['17'].conv._modules['6'].weight = finenet.module.features._modules['17'].conv._modules['6'].weight
    net.module.features._modules['17'].conv._modules['7'].weight = finenet.module.features._modules['17'].conv._modules['7'].weight
    net.module.features._modules['17'].conv._modules['7'].bias = finenet.module.features._modules['17'].conv._modules['7'].bias
    net.module.features._modules['17'].conv._modules['7'].running_mean = finenet.module.features._modules['17'].conv._modules['7'].running_mean
    net.module.features._modules['17'].conv._modules['7'].running_var = finenet.module.features._modules['17'].conv._modules['7'].running_var
    net.module.features._modules['18']._modules['0'].weight = finenet.module.features._modules['18']._modules['0'].weight
    net.module.features._modules['18']._modules['1'].weight = finenet.module.features._modules['18']._modules['1'].weight
    net.module.features._modules['18']._modules['1'].bias = finenet.module.features._modules['18']._modules['1'].bias
    net.module.features._modules['18']._modules['1'].running_mean = finenet.module.features._modules['18']._modules['1'].running_mean
    net.module.features._modules['18']._modules['1'].running_var = finenet.module.features._modules['18']._modules['1'].running_var
    return net

