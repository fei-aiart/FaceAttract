import torch
import torch.nn as nn
import math
import torchvision
from Mv2_chaatt import *
from Mv2attn import *

def parsing_net1():
	parsing_net1 =MV2_cattn()
	parsing_net1 = torch.nn.DataParallel(parsing_net1).cuda()
	#parsing_net1.module.classifier._modules['1'] =  nn.Linear(1280, 1)
	#print 'parsing_net1----',parsing_net1
	#parsing_net1.load_state_dict(torch.load('/data5/xuantong/pytorch/f5/mv2/parsingatt/11_channel/checkpoint/net_epoch_20.weight'))
	parsing_net1.module.conv = nn.Conv2d(7, 32, 3, 2, 1, bias=False)
	parsing_net1.module.fc = nn.Linear(7,7)
	parsing_net1.module.classifier._modules['1'] =  nn.Linear(1280, 1280)
	#print 'parsing_net2----',parsing_net1
	return parsing_net1
def image_net1():
	image_net1 = MV2attn()
	image_net1 = torch.nn.DataParallel(image_net1).cuda()
	#image_net1.load_state_dict(torch.load('/data5/xuantong/pytorch/f5/mv2/align/2/checkpoint1/net_epoch_41.weight'))
	image_net1.module.attention.L3 = nn.Linear(in_features=1280,out_features=1280)
	return image_net1

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.bilinear = nn.Bilinear(256,256,1)
		self.parsing_net1 = parsing_net1()
		self.image_net1 = image_net1()
		self.fc = nn.Linear(2560, 1280)
		self.fc1 = nn.Linear(1280, 1)
		#self.fc = nn.Linear(1024, 5)

		#self.fc = nn.Linear(2656, 1)


	def forward(self,image,parsing,inputs2):
		x1 = self.parsing_net1(parsing,inputs2)
		#print 'x1',x1.size()
		x2 = self.image_net1(image)
		#print 'x2',x2.size()
		x = torch.cat((x1,x2),1)
		#print 'x',x.size()
		#x = self.bilinear(x1,x2)
		# x = x1+x2
		x = self.fc(x)
		x = self.fc1(x)
		return x




