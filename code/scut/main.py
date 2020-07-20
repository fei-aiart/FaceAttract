# -*- coding: utf-8 -*-
# @Author: JacobShi777


import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
#import cv2
import random
import argparse
import random
import functools
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data import *
from MobileNetV2 import *
from Mv2_chaatt import *
from net import *
import option
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

opt = option.init()
f3 = open('./checkpoint/lr.txt', 'w')
# optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)

def train():	
	# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
	f = open('./checkpoint/loss.txt', 'w')
	f2 = open('./checkpoint/log.txt', 'w')
	#net = InceptionV4(num_classes=2)
	checkpaths(opt)

	train_set = DatasetFromFolder(opt, True,False)
	train_set2 = DatasetFromFolder(opt, True,True)
	test_set = DatasetFromFolder(opt, False,False)
	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
	training_data_loader2 = DataLoader(dataset=train_set2, num_workers=opt.threads, batch_size=1, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=True)

	# net1 = MobileNetV2()
	# net1 = torch.nn.DataParallel(net1).cuda()
	# net1.load_state_dict(torch.load('/data5/xuantong/pytorch/nima/MobileNet-V2-Pytorch-master/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'))

	# net = MV2_cattn()
	# net = torch.nn.DataParallel(net).cuda()
	# net = load_weights(net,net1)
	# #print 'static',net.state_dict().keys()
	# net.module.classifier._modules['1'] =  nn.Linear(1280, 1)
	net = Net()
	#print net
	if opt.cuda:
		net = net.cuda()
	#criterion = torch.nn.CrossEntropyLoss()
	#criterion = torch.nn.L1Loss()
	criterion = torch.nn.MSELoss()
	#criterion = torch.nn.SmoothL1Loss()
	#optimizer = torch.optim.SGD(
		#[{'params':net.fc.parameters()}],lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

		#net.parameters(),lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
	optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)
	start_time = time.time()
	for epoch in range(1, opt.n_epoch+1):
		adjust_learning_rate(optimizer, epoch)
		loss_running = 0.0
		for (i, batch) in enumerate(training_data_loader, 1):
			#test(net, epoch, testing_data_loader, opt, f2)
			# img,mat, targets = Variable(batch[0]),Variable(batch[1]) ,Variable(batch[2])
			# inputs2 = torch.autograd.Variable((torch.randn(int(mat.shape[0]),7)))
			img, mat, targets = batch[0], batch[1], batch[2]
			inputs2 = torch.randn(int(mat.shape[0]), 7)
			if opt.cuda:
				img,mat, targets,inputs2 = img.cuda(),mat.cuda(), targets.cuda(), inputs2.cuda()

			net.train()
			#print 'inputs',inputs.shape

			optimizer.zero_grad()
			outputs = net(img,mat,inputs2)
			#print 'targets',targets.size()
			#print 'outputs',outputs.size()
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()

			loss_running += loss.item()

			if i % opt.print_every == 0:
				end_time = time.time()
				time_delta = usedtime(start_time, end_time)
				print('[%s-%d, %6d] loss: %.3f' %(time_delta, epoch, i*opt.batchSize, loss_running/opt.print_every))
				f.write('%d, %d, loss:%.3f\r\n' %(epoch, i*opt.batchSize, loss_running/opt.print_every))
				loss_running = 0.0

			# if (i)%300==0:
			# 	test(net, epoch, testing_data_loader, opt, f2)
			# 	# net.dropout = nn.Dropout(p=0.75)
			# 	net.train()

		f.flush()

		if (epoch)%opt.test_period==0:
			test(net, epoch, testing_data_loader, opt, f2)
			if (epoch)%1==0:
				checkpoint(epoch, net)
	f.close()

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
	lr = opt.lr * (0.5 ** (epoch // 8))
	# log to TensorBoard
	#if args.tensorboard:
	#   log_value('learning_rate', lr, epoch)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	print(lr)
	f3.write('lr:%f\r\n' %(lr))
	f3.flush()

def test(net, epoch, testing_data_loader, opt, f2):
	pre_score = []
	true_score = []
	net.eval()
	counter = 0
	counter_one = 0
	test_loss = 0.0
	for (i, batch) in enumerate(testing_data_loader, 1):
		#criterion = torch.nn.L1Loss()
		#criterion = torch.nn.SmoothL1Loss()
		criterion = torch.nn.MSELoss()
		img, mat, targets = batch[0], batch[1], batch[2]
		inputs2 = torch.randn(int(mat.shape[0]), 7)
		if opt.cuda:
			img,mat, targets,inputs2 = img.cuda(),mat.cuda(), targets.cuda(), inputs2.cuda()
		outputs = net(img,mat,inputs2)
		loss = criterion(outputs, targets)
		outputs = outputs.data.cpu().numpy()[0][0]
		targets = targets.data.cpu().numpy()[0][0]
		test_loss += loss.item()
		pre_score.append(outputs)
		true_score.append(targets)
	plcc=pearsonr(pre_score,true_score)
	srcc=spearmanr(pre_score,true_score)
	krcc=kendalltau(pre_score,true_score)
	print('plcc:',plcc[0])
	print('srcc:',srcc[0])
	print('krcc:',krcc[0])
	f2.write('%d,test_loss: %.3f, plcc: %.3f, srcc: %.3f, krcc: %.3f\r\n' %(epoch,test_loss/i,plcc[0],srcc[0],krcc[0]))
	f2.flush()
	print('test_loss: %.3f' %( test_loss/i))
	# print counter_one
if __name__=='__main__':
	train()
	f3.close()
