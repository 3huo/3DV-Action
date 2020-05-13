# -*- coding: utf-8 -*-
import torch
import os
import tqdm
import shutil
import collections
import argparse
import random
import time
import logging
import numpy as np

from model import PointNet_Plus#,Attension_Point,TVLAD
from dataset import NTU_RGBD
from utils import group_points,group_points_pro

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def main(args=None):
	parser = argparse.ArgumentParser(description = "Training")

	parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
	parser.add_argument('--nepoch', type=int, default=80, help='number of epochs to train for')
	parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 8,  help='number of input point features')
	parser.add_argument('--temperal_num', type=int, default = 5,  help='number of input point features')
	parser.add_argument('--pooling', type=str, default='concatenation', help='how to aggregate temporal split features: vlad | concatenation | bilinear')
	parser.add_argument('--dataset', type=str, default='ntu60', help='how to aggregate temporal split features: ntu120 | ntu60')


	parser.add_argument('--weight_decay', type=float, default=0.0008, help='weight decay (SGD only)')
	parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate at t=0')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')
	parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

	parser.add_argument('--root_path', type=str, default='/UCLA_point/UCLA_vsize40_feature_2048_ff_rawdi_2',  help='preprocess folder')# 3DV points path
	parser.add_argument('--depth_path', type=str, default='3DV_construction/UCLA_point/UCLA_bboxed_2048',  help='raw_depth_png')# appearance points path
	parser.add_argument('--save_root_dir', type=str, default='results_ucla/UCLA_v40_MultiStream_rawdi2',  help='output folder')
	parser.add_argument('--model', type=str, default = '',  help='model name for training resume')
	parser.add_argument('--optimizer', type=str, default = '',  help='optimizer name for training resume')
	parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
	parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

	
	parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

	parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
	parser.add_argument('--SAMPLE_NUM', type=int, default = 2048,  help='number of sample points')

	parser.add_argument('--Num_Class', type=int, default = 10,  help='number of outputs')
	parser.add_argument('--knn_K', type=int, default = 64,  help='K for knn search')
	parser.add_argument('--sample_num_level1', type=int, default = 512,  help='number of first layer groups')
	parser.add_argument('--sample_num_level2', type=int, default = 128,  help='number of second layer groups')
	parser.add_argument('--ball_radius', type=float, default=0.11, help='square of radius for ball query in level 1')#0.025 -> 0.05 for detph
	parser.add_argument('--ball_radius2', type=float, default=0.24, help='square of radius for ball query in level 2')# 0.08 -> 0.01 for depth


	opt = parser.parse_args()
	print (opt)



	torch.cuda.set_device(opt.main_gpu)

	opt.manualSeed = 1
	random.seed(opt.manualSeed)
	torch.manual_seed(opt.manualSeed)

	try:
		os.makedirs(opt.save_root_dir)
	except OSError:
		pass
	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(opt.save_root_dir, 'print.log'), level=logging.INFO)
	logging.info('======================================================')

	os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

	torch.backends.cudnn.benchmark = True
	#torch.backends.cudnn.deterministic = True
	torch.cuda.empty_cache()

	data_train = NTU_RGBD(root_path = opt.root_path,opt=opt,
		test = False,
		Transform = True
		)
	train_loader = DataLoader(dataset = data_train, batch_size = opt.batchSize, shuffle = True, drop_last = True)
	data_val = NTU_RGBD(root_path = opt.root_path, opt=opt,
		test = True,
		Transform = False
		)
	val_loader = DataLoader(dataset = data_val, batch_size = 24)

	#net =

	netR = PointNet_Plus(opt)
	#netR = Attension_Point(opt)
	#netR = load_state_dict(torch.load())
	netR.load_state_dict(torch.load('pointnet_para_44.pth'),strict = False)
	netR = torch.nn.DataParallel(netR).cuda()
	netR.cuda()
	print(netR)

	criterion = torch.nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.Adam(netR.parameters(), lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


	for epoch in range(opt.nepoch):
		scheduler.step(epoch)
		
		# switch to train mode
		torch.cuda.synchronize()
		netR.train()
		acc = 0.0
		loss_sigma = 0.0
		total1 = 0.0
		timer = time.time()

		for i, data in enumerate(tqdm(train_loader, 0)):
			if len(data[0])==1:
				continue
			torch.cuda.synchronize()
			#print(i)
			# 1 load imputs and target
			points_xyzc,points_1xyz,points2_xyz,points3_xyz,label,vid_name = data
			#if i == 1:
			#print(vid_name)
			points_xyzc,points_1xyz,points2_xyz,points3_xyz,label = points_xyzc.cuda(),points_1xyz.cuda(),points2_xyz.cuda(),points3_xyz.cuda(),label.cuda()
			#print(points_xyz.shape)
			#print(label,vid_name)
			# points: B*2048*4; target: B*1
			opt.ball_radius = opt.ball_radius + random.uniform(-0.02,0.02)
			xt, yt = group_points_pro(points_xyzc, opt)

			xs1, ys1 = group_points(points_1xyz, opt)
			xs2, ys2 = group_points(points2_xyz, opt)
			xs3, ys3 = group_points(points3_xyz, opt)
			#print('123',inputs_level1_r[1,:,55])
			#print(inputs_level1_r.shape,'123',inputs_level1_r[1,:,88,55])
			# 2 compute outputs:
			prediction = netR(xt, xs1, xs2, xs3, yt, ys1, ys2, ys3)
			#print(pridiction)
			loss = criterion(prediction,label)

			optimizer.zero_grad()

			# compute gradient and do SGD step
			loss.backward()
			optimizer.step()
			torch.cuda.synchronize()

			# update training error
			loss_sigma += loss.item()
			#print(prediction.data.shape)
			#_, predicted60 = torch.max(prediction.data[:,0:60], 1)
			_, predicted = torch.max(prediction.data, 1)
			#print(predicted,'----',label)
			#print(predicted.shape,predicted60)
			#print(predicted,predicted60)
			#print(predicted==label)
			#print(label,'label')
			acc += (predicted==label).cpu().sum().numpy()
			total1 += label.size(0)

		
		acc_avg = acc/total1
		loss_avg = loss_sigma/total1
		print('======>>>>> Online epoch: #%d, lr=%f,Acc=%f,avg_loss=%f  <<<<<======' %(epoch, scheduler.get_lr()[0],acc_avg,loss_avg))
		#print("Epoch: " + str(epoch) + " Iter: " + str(i) + " Acc: " + ("%.2f" % acc_avg) +" Classification Loss: " + str(loss_avg))
		 
		if epoch>-1:
			# evaluate mode
			torch.cuda.synchronize()
			netR.eval()
			conf_mat = np.zeros([opt.Num_Class, opt.Num_Class])
			acc = 0.0
			loss_sigma = 0.0


			for i, data in enumerate(tqdm(val_loader)):
				#print(i)
				if i >-1:
					torch.cuda.synchronize()
					points_xyzc,points_1xyz,points2_xyz,points3_xyz,label,vid_name = data
					#if i == 1:
					#print(vid_name)
					points_xyzc,points_1xyz,points2_xyz,points3_xyz,label = points_xyzc.cuda(),points_1xyz.cuda(),points2_xyz.cuda(),points3_xyz.cuda(),label.cuda()
					#print(points_xyz.shape)
					#print(label,vid_name)
					# points: B*2048*4; target: B*1
					opt.ball_radius = opt.ball_radius + random.uniform(-0.02,0.02)
					xt, yt = group_points_pro(points_xyzc, opt)

					xs1, ys1 = group_points(points_1xyz, opt)
					xs2, ys2 = group_points(points2_xyz, opt)
					xs3, ys3 = group_points(points3_xyz, opt)
					#print('123',inputs_level1_r[1,:,55])
					#print(inputs_level1_r.shape,'123',inputs_level1_r[1,:,88,55])
					# 2 compute outputs:
					prediction = netR(xt, xs1, xs2, xs3, yt, ys1, ys2, ys3)
					#print(pridiction)
					loss = criterion(prediction,label)
					_, predicted = torch.max(prediction.data, 1)
					#print(prediction.data)
					loss_sigma += loss.item()

					for j in range(len(label)):
						cate_i = label[j].cpu().numpy()
						#print(cate_i)
						pre_i = predicted[j].cpu().numpy()
						conf_mat[cate_i, pre_i] += 1.0


			print('UCLA:{:.2%}  ===Average loss:{:.6%}'.format(conf_mat.trace() / conf_mat.sum(),loss_sigma/(i+1)/16))
		logging.info('{} --epoch{} UCLA:{:.2%}  ===Average loss:{:.6%}'.format('Valid',epoch,conf_mat.trace() / conf_mat.sum(),loss_sigma/(i+1)/16))

		torch.save(netR.module.state_dict(), '%s/pointnet_para_%d.pth' % (opt.save_root_dir, epoch))
if __name__ == '__main__':
	main()

