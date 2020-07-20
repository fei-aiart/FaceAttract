# -*- coding: utf-8 -*-
# @Author: JacobShi777

import cv2
import os
import random
import numpy as np
import torch
import random
import scipy.io as sio
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, RandomCrop, CenterCrop, Normalize,RandomHorizontalFlip



def formnames(infofile, if_train,if_test):
	if if_train == True:
		if if_test == True:
			infofile = infofile[2]
		else:
			infofile = infofile[0]
	else:
		infofile = infofile[1]

	#infofile = infofile[0] if if_train else infofile[1]
	res = []
	with open(infofile, 'r') as f:
		for line in f:
			res.append(line.strip())
	return res

def input_transform(if_train, opt):
	if if_train:
		# transform = transforms.Compose([RandomCrop(opt.inputSize), ToTensor(), \
		# 	Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
		transform = transforms.Compose([ToTensor(),])
		#transform = transforms.Compose([RandomHorizontalFlip(),RandomCrop(opt.cropSize),ToTensor(),Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	else:
		# transform = transforms.Compose([CenterCrop(opt.inputSize), ToTensor(), \
		# 	Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
		transform = transforms.Compose([ToTensor(),])
		#transform = transforms.Compose([RandomHorizontalFlip(),RandomCrop(opt.cropSize),ToTensor(), Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	return transform


def mat_process(img_fl):
	# img_fl = np.transpose(img_fl, (2, 1, 0))
	img_fl = img_fl.astype(np.float32)
	img = img_fl[0:2, :, :]
	temp = img_fl[2:, :, :]
	# print temp.shape
	l0 = temp[0, :, :] + temp[1, :, :]
	l0 = np.where(l0 > 1, 1, l0)

	l1 = temp[2, :, :] + temp[3, :, :]
	l1 = np.where(l1 > 1, 1, l1)

	l2 = temp[8, :, :]

	l3 = temp[9, :, :] + temp[10, :, :] + temp[11, :, :]
	l3 = np.where(l3 > 1, 1, l3)
	# l6 = temp[8,:,:]

	l4 = temp[15, :, :]

	# merge
	img = np.concatenate((img, l0.reshape(1, l0.shape[0], l0.shape[1])), axis=0)
	img = np.concatenate((img, l1.reshape(1, l1.shape[0], l1.shape[1])), axis=0)
	img = np.concatenate((img, l2.reshape(1, l2.shape[0], l2.shape[1])), axis=0)
	img = np.concatenate((img, l3.reshape(1, l3.shape[0], l3.shape[1])), axis=0)
	img = np.concatenate((img, l4.reshape(1, l4.shape[0], l4.shape[1])), axis=0)
	# img = np.transpose(img, (2, 1, 0))
	return img



def load_inputs(matname, opt, if_train):
	root = opt.root[0] if if_train else opt.root[1]
	name = matname.split('.')
	imgname = name[0]+'.jpg'
	imgpath = os.path.join(root,'scut_align/', imgname)
	#img = cv2.imread(imgpath)
	img = Image.open(imgpath)
	#print  'img' ,img
	transform =  transforms.Compose([RandomHorizontalFlip(),ToTensor(),Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
	img = transform(img)
	matname = 'scut/'+name[0]+'.npy'
	# matpath = os.path.join(root, matname)
	# mat = sio.loadmat(matpath)['res_label']
	mat = np.load(os.path.join(root, matname))
	mat = mat_process(mat).transpose(1,2,0)
	img = np.transpose(img.numpy(), (1,2,0))
	#img = np.concatenate((img, mat), axis=2)
	img = cv2.resize(img, (opt.cropSize, opt.cropSize), interpolation=cv2.INTER_CUBIC)
	w_offset = random.randint(0, opt.resizeSize - opt.cropSize - 1)
	h_offset = random.randint(0, opt.resizeSize - opt.cropSize- 1)
	# img = img[w_offset:w_offset+opt.cropSize, h_offset:h_offset+opt.cropSize,:]
	#print 'img',img
	mat = cv2.resize(mat, (opt.resizeSize, opt.resizeSize), interpolation=cv2.INTER_CUBIC)
	mat = mat[w_offset:w_offset+opt.cropSize, h_offset:h_offset+opt.cropSize,:]
	return img,mat

class DatasetFromFolder(data.Dataset):
	def __init__(self, opt, if_train,if_test):
		super(DatasetFromFolder, self).__init__()

		self.if_train = if_train
		self.if_test = if_test
		self.opt = opt
		self.infonames = formnames(opt.infofile, self.if_train,self.if_test)
		self.input_transform = input_transform(self.if_train, self.opt)

	def __getitem__(self, index):
		infoname = self.infonames[index]
		item = infoname.split(' ')
		inputs = load_inputs(item[0], self.opt, self.if_train)
		#print 'inputs----',inputss
		#inputs = self.input_transform(inputs)
		img = inputs[0]
		mat = inputs[1]
		#print 'img',img.shape
		#print 'mat',mat.shape
		img = self.input_transform(img)
		mat = self.input_transform(mat)
		#targets = torch.LongTensor([int(item[1])])
		targets = torch.FloatTensor([float(item[1])])

		#print 'targets---',targets
		#print inputs
		# print targets
		return img,mat, targets

	def __len__(self):
		return len(self.infonames)


def checkpaths(opt):
	if not os.path.exists('./checkpoint'):
		os.mkdir('./checkpoint')


def checkpoint(epoch, net):
	net_out_path = "./checkpoint/net_epoch_{}.weight".format(epoch)
	torch.save(net.state_dict(), net_out_path)
	print("Checkpoint saved to {}".format(net_out_path))


def usedtime(strat_time, end_time):
	delta = int(end_time - strat_time)
	hours = delta // 3600
	minutes = (delta-hours*3600)//60
	seconds = delta-hours*3600-minutes*60
	return ('%2d:%2d:%2d' %(hours,minutes,seconds))


