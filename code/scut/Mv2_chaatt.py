import torch.nn as nn
import math

###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def conv_bn(oup):
    return nn.Sequential(
        #nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp,oup):
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


class MV2_cattn(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MV2_cattn, self).__init__()
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
        #net.module.features._modules['0']._modules['0']
        self.conv = nn.Conv2d(7, input_channel, 3, 2, 1, bias=False)
        self.features = [conv_bn(input_channel)]
        # building inverted residual blocks
        #
        #self.BatchNorm = nn.BatchNorm2d(32)
        #self.relu = nn.ReLU6(inplace=True)
        self.fc = nn.Linear(11,11)
        self.softmax = nn.Softmax(dim=1)


                                                                                                             
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
        self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )


    def forward(self, x,x1):
        
        #x = self.BatchNorm(x)
        #x = self.relu(x)
        #print 'xx',x.size()
        x1 = self.fc(x1)
        x1 = self.softmax(x1)
        x1 = x1.unsqueeze(2).unsqueeze(3)
        #print 'x1',x1.size()
        x = x1*x
        #print 'x',x.size()
        x = self.conv(x)
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x



def load_weights(net, finenet):
    # net.module.features._modules['0']._modules['0'].weight = finenet.module.features._modules['0']._modules['0'].weight
    # net.module.features._modules['0']._modules['1'].weight = finenet.module.features._modules['0']._modules['1'].weight
    # net.module.features._modules['0']._modules['1'].bias = finenet.module.features._modules['0']._modules['1'].bias
    # net.module.features._modules['0']._modules['1'].running_mean = finenet.module.features._modules['0']._modules['1'].running_mean
    # net.module.features._modules['0']._modules['1'].running_var = finenet.module.features._modules['0']._modules['1'].running_var
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