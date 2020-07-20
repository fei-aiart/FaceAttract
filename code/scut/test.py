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

f2 = open('./checkpoint/log.txt', 'w')
def test(num):
	opt = option.init()
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	pre_list = []
	net = Net()
	#net = torch.nn.DataParallel(net).cuda()	
	net.load_state_dict(torch.load(opt.pretrain))
	#K:/data5/xuantong/pytorch/nima/mobile_net/emd_plcc/4/checkpoint1/
	net = net.cuda()

	mae = 0.
	rmse = 0.

	for k in range(0,num):
		true_score = []
		pre_score = []
		print('test_{}'.format(k)	)
		test_set = DatasetFromFolder(opt, False,False)
		testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
		net.eval()
		loss_running = 0.0

		maeSum = 0.
		rmseSum = 0.

		for (i, batch) in enumerate(testing_data_loader, 1):
			#criterion = torch.nn.CrossEntropyLoss()
			criterion = torch.nn.MSELoss()
			# img,mat, targets = Variable(batch[0]),Variable(batch[1]) ,Variable(batch[2])
			# inputs2 = torch.autograd.Variable((torch.randn(int(mat.shape[0]),7)))
			img, mat, targets = batch[0], batch[1], batch[2]
			inputs2 = torch.randn(int(mat.shape[0]), 7)
			if opt.cuda:
				img,mat, targets,inputs2 = img.cuda(),mat.cuda(), targets.cuda(), inputs2.cuda()
			#print 'test_targets--',targets
			outputs = net(img,mat,inputs2)
			#print 'outputs',outputs
			#print 'test_outputs--',outputs
			loss = criterion(outputs, targets)
			loss_running += loss.item()

			pred = outputs.data.cpu().numpy()
			#print 'pred',pred.shape
			true = targets.data.cpu().numpy()

			maeSum += np.abs(pred - true)
			rmseSum += np.power((pred - true), 2)

			#print's---',int(true.shape[0])
			for j in range(0,int(true.shape[0])):
			#print 'true',true.shape
				true_score.append(true[j])
				pre_score.append(pred[j])
			#print'true',true.size

			if i % opt.print_every == 0:
				end_time = time.time()
				print('[%d, %6d] loss: %.3f' %(k ,i*opt.batchSize, loss_running/opt.print_every))
				f2.write('%d, %d, loss:%.3f\r\n' %(k, i*opt.batchSize, loss_running/opt.print_every))
				loss_running = 0.0

		mae += (maeSum/i)
		rmse += np.sqrt(rmseSum/i)

		np.save('./checkpoint/pre_score_{}.npy'.format(k),pre_score)
		pre_list.append(pre_score)
		print('pre_list',len(pre_list))

	print(mae / 10, rmse / 10)
	
	np.save('./checkpoint/true_score.npy',true_score)		
	np.save('./checkpoint/pre_list.npy',pre_list)
	pre = np.array(pre_list)
	#pre_max = pre.max(0)
	#pre_min =pre.min(0) 
	#pre_sum = pre.sum(0)
#%%
	#pre2 = pre_sum-pre_min-pre_max
#%%
	#mm = int(pre.shape[0])-2
	#pre_avg = pre2/mm
	pre_avg = pre.mean(0)
	plcc=pearsonr(pre_avg,true_score)
	srcc=spearmanr(pre_avg,true_score)
	krcc=kendalltau(pre_avg,true_score)	
	print('plcc:',plcc[0])
	print('srcc:',srcc[0])
	print('krcc:',krcc[0])
	f2.write('%d, plcc: %.3f, srcc: %.3f, krcc: %.3f \r\n' %(j,plcc[0],srcc[0],krcc[0]))
	f2.flush()
if __name__=='__main__':
	test(10)
	f2.close()


