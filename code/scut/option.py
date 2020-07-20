# -*- coding: utf-8 -*-
# @Author: JacobShi777

import argparse

def init():
	parser = argparse.ArgumentParser(description='PyTorch')
	parser.add_argument('--root', default=["/home/meimei/mayme/data/", \
		"/home/meimei/mayme/data/"], help='image source folder')
	parser.add_argument('--checkpoint', type=str, default='./checkpoint', help='checkpoint folder')
	parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
	parser.add_argument('--print_every', type=int, default=10, help='print every time')
	parser.add_argument('--threads', type=int, default=16, help='number of threads for data5 loader to use')
	parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
	# parser.add_argument('--gpu_ids', default=[0], help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	parser.add_argument('--gpu_ids', default="3", help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
	# parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
	parser.add_argument('--cuda',  default=True, help='use cuda?')
	parser.add_argument('--n_epoch', type=int, default=100, help='training epoch')
	parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
	parser.add_argument('--resizeSize', type=int, default=256, help='then resize to this size')
	parser.add_argument('--infofile', default=['./data1/mat_train_1.txt', './data1/mat_test_1.txt','./data1/mat_test_1.txt'], help='infofile')
	parser.add_argument('--batchSize', type=int, default=50, help='training batchSize')
	parser.add_argument('--test_period', type=int, default=1, help='test_period')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
	parser.add_argument('--weight_decay', default=1e-5, type=float,help='weight decay')
	parser.add_argument('--pretrain', default='../pretrain_model/net_cross_1.weight', type=float,help='weight decay')

	opt = parser.parse_args()
	return opt