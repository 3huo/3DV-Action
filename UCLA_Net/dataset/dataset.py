import os
import tqdm
import torch
import re
import collections
import imageio
import random

from tqdm import tqdm
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import scipy.io as sio


compiled_regex = re.compile('.*a(\d{2})_s(\d{2})_e(\d{2})_v(\d{2}).*')
SAMPLE_NUM = 2048

class NTU_RGBD(Dataset):
	"""NTU depth human masked datasets"""
	def __init__(self, root_path, opt,
		test=False, 
		Transform = True):

		self.root_path = root_path
		self.depth_path = opt.depth_path
		self.SAMPLE_NUM = opt.SAMPLE_NUM
		self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
		self.transform = Transform
		self.depth_path = opt.depth_path

		self.point_vids = os.listdir(self.root_path)#.sort()
		self.point_vids.sort()
		#print(self.point_vids)
		self.num_clouds = len(self.point_vids)
		print(self.num_clouds)
		self.point_data   = self.load_data()

		
		self.set_splits()


		self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
		self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

		if test: self.vid_ids = self.test_split_camera.copy()
		else: self.vid_ids = self.train_split_camera.copy()

		print('num_data:',len(self.vid_ids))


		self.SAMPLE_NUM = opt.SAMPLE_NUM
		self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM

		self.point_clouds = np.empty(shape=[self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],dtype=np.float32)

	def __getitem__(self, idx):
		vid_id = self.vid_ids[idx]
		vid_name = self.id_to_vidName[vid_id]

		v_name = vid_name[:-9]
		#print(v_name)

		
		path_depth = self.depth_path + '/' +v_name
		points_list = os.listdir(path_depth)
		len_points = len(points_list)
		points_2048 = np.zeros((3,2048,3))
		for i in range(3):
			if len_points>3:
				seg_length = round(len_points/3)
				random_index = seg_length*i + random.randint(1,seg_length)
				if random_index>len_points:
					random_index = len_points
	
				frame_name = points_list[random_index-1]
				#print(v_name,frame_name)
				path_points_frame = os.path.join(path_depth,frame_name)
				#print(path_points_frame)
				points = sio.loadmat(path_points_frame)
				points_2048[i] = points['pc'].astype(np.float32)
			else:
				print("error bbox number!")


		path_cloud_npy = os.path.join(self.root_path,self.id_to_vidName[vid_id])
		#points_c = XYZ_C['pc'].astype(np.float32)
		#points_c = np.load(path_cloud_npy)
		XYZ_C = sio.loadmat(path_cloud_npy)
		#print(self.id_to_vidName[vid_id])
		points_c = XYZ_C['pc'].astype(np.float32)
		label = self.id_to_action[vid_id]
		points_c = np.expand_dims(points_c, axis=0)
		theta = np.random.rand()*2-1
		#if self.DATA_CROSS_VIEW == False:
			#theta = theta/4
		#print('<<',points_2048[0])
		if self.transform:
			points_c, points_2048 = self.point_transform(points_c,points_2048,theta)
		points_2048 = torch.tensor(points_2048,dtype=torch.float)
		label = torch.tensor(label)
		#print(points_2048[0],'>>')
		#print('----',points_2048)
		return points_c,points_2048[0],points_2048[1],points_2048[2],label,vid_name

	def __len__(self):
		return len(self.vid_ids)



	def load_data(self):
		self.point_data = []
		for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
			self.point_data.append(self.get_pointdata(cloud_idx))

		return self.point_data

	def get_pointdata(self, vid_id):


		vid_name = self.point_vids[vid_id]
		#print(vid_name)
		match = re.match(compiled_regex, vid_name)
		action, setup, eeer, view = [*map(int, match.groups())]
		if action ==11:
			action = 7
		if action == 12:
			action = 10
		return {
			'video_cloud_name': vid_name,
			'video_index': vid_id,
			'setup':       setup,
			'eeer':        eeer,
			'view':        view,
			'action':      action,
		}


	def set_splits(self):
		'''
		Sets the train/test splits
		Cross-Subject Evaluation:
			Train ids = 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27,
					28, 31, 34, 35, 38
		Cross-View Evaluation:
			Train camera views: 2, 3
		'''
		# Save the dataset as a dataframe
		dataset = pd.DataFrame(self.point_data)
		# Get the train split ids
		train_ids_view  = [1, 2]
		# Cross-View splits
		self.train_split_camera  = list(
			dataset[dataset.view.isin(train_ids_view)]['video_index'])
		self.test_split_camera  = list(
			dataset[~dataset.view.isin(train_ids_view)]['video_index'])


	def point_transform(self,points_c,points_xyz,y):

		# input: temporal_num*2048*3
		#R_y = torch.stack([
		#	torch.stack([torch.cos(y), torch.tensor(0.), torch.sin(y)]),
		#	torch.tensor([0., 1., 0.]),
		#	torch.stack([-torch.sin(y), torch.tensor(0.), torch.cos(y)])
		#])
		#points = torch.mm(points,R_y)
		anglesX = (np.random.uniform()-0.5) * (1/9) * np.pi
		R_y = np.array([[[np.cos(y),0.0,np.sin(y)],
			[0.0,1.0,0.0],
			[-np.sin(y),0.0,np.cos(y)]]])
		R_x = np.array([[[1, 0, 0],
			[0, np.cos(anglesX), -np.sin(anglesX)],
 			[0, np.sin(anglesX), np.cos(anglesX)]]])

		points_c[:,:,0:3] = self.jitter_point_cloud(points_c[:,:,0:3],sigma=0.007, clip=0.04)
		points_xyz = self.jitter_point_cloud(points_xyz,sigma=0.012, clip=0.006)

		points_c[:,512:SAMPLE_NUM,:] = self.random_dropout_point_cloud(points_c[:,512:SAMPLE_NUM,:])
		points_xyz[:,512:SAMPLE_NUM,:] = self.random_dropout_point_cloud(points_xyz[:,512:SAMPLE_NUM,:])

		R =  np.matmul(R_y, R_x)

		#print(points.shape)
		# for i in range(5):
		# 	cc = points[:,3+i]
		# 	if np.random.rand()>0.8:
		# 		scale = np.random.rand()-0.5
		# 		cc[cc>0] = np.power(cc[cc>0],1+scale/3)
		# 	if np.random.rand()>0.8:
		# 		scale = np.random.rand()-0.5	
		# 		cc[cc<0] = -np.power(-cc[cc<0],1+scale/3)
		
		points_c[:,:,0:3] = np.matmul(points_c[:,:,0:3],R)
		points_xyz = np.matmul(points_xyz,R)


		if np.random.rand()>0.6:
			for i in range(3):
				points_c[:,:,i] = points_c[:,:,i]+(np.random.rand()-0.5)/6
				points_xyz[:,:,i] = points_xyz[:,:,i]+(np.random.rand()-0.5)/6
		
		#print(points.shape)
		return points_c, points_xyz


	def load_depth_from_img(self,depth_path):
		depth_im = imageio.imread(depth_path) #im is a numpy array
		return depth_im

	def depth_to_pointcloud(self,depth_im):
		rows,cols = depth_im.shape
		xx,yy = np.meshgrid(range(0,cols), range(0,rows))

		valid = depth_im > 0
		xx = xx[valid]
		yy = yy[valid]
		depth_im = depth_im[valid]

		X = (xx - cx) * depth_im / fx
		Y = (yy - cy) * depth_im / fy
		Z = depth_im

		points3d = np.array([X.flatten(), Y.flatten(), Z.flatten()])

		return points3d

	def farthest_point_sampling_fast(self, pc, sample_num):
		pc_num = pc.shape[0]

		sample_idx = np.zeros(shape = [sample_num,1], dtype = np.int32)
		sample_idx[0] = np.random.randint(0,pc_num)

		cur_sample = np.tile(pc[sample_idx[0],:], (pc_num,1))
		diff = pc-cur_sample
		#print(sample_idx[0])
		min_dist = (diff*diff).sum(axis = 1)
		#
		#print(min_dist.shape,diff.shape,pc.shape)
		for cur_sample_idx in range(1,sample_num):
			## find the farthest point

			sample_idx[cur_sample_idx] = np.argmax(min_dist)
			#print(cur_sample_idx,np.argmax(min_dist))
			if cur_sample_idx < sample_num-1:
				#valid_idx = np.where(min_dist > 1e-5)
				#valid_idx = valid_idx[0]
				#print(valid_idx.shape)
				diff = pc - np.tile(pc[sample_idx[cur_sample_idx],:], (pc_num,1))
				#print(diff.shape,min_dist[valid_idx].shape)
				min_dist = np.concatenate((min_dist.reshape(pc_num,1), (diff*diff).sum(axis = 1).reshape(pc_num,1)), axis = 1).min(axis = 1)  ##?
		#print(min_dist)
		return sample_idx

	def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
		"""

		:param data: Nx3 array
		:return: jittered_data: Nx3 array
		"""
		M, N, C = data.shape
		jittered_data = np.clip(sigma * np.random.randn(M, N, C), -1 * clip, clip).astype(np.float32)

		jittered_data = data+jittered_data

		return jittered_data

	def random_dropout_point_cloud(self, data):
		"""

		:param data:  Nx3 array
		:return: dropout_data:  Nx3 array
		"""
		M, N, C = data.shape
		dropout_ratio = 0.7+ np.random.random()/2
		#dropout_ratio = np.random.random() * p
		drop_idx = np.where(np.random.random(N) <= dropout_ratio)[0]
		dropout_data = np.zeros_like(data)
		if len(drop_idx) > 0:
			dropout_data[:, drop_idx, :] = data[:, drop_idx, :]


		# xyz_center = np.random.random(3)
		# xyz_d = 0.1+np.random.random(3)/10

		# func_x = lambda d: d>xyz_center[0] and d<(xyz_center[0]+xyz_d[0])
		# func_y = lambda d: d>xyz_center[1] and d<(xyz_center[1]+xyz_d[1])
		# func_z = lambda d: d>xyz_center[2] and d<(xyz_center[2]+xyz_d[2])
		# c_x = np.vectorize(func_x)(data[:,0])
		# c_y = np.vectorize(func_x)(data[:,0])
		# c_z = np.vectorize(func_x)(data[:,0])
		# c = c_x*c_z*c_y
		# erase_index = np.where(c)
		# dropout_data[erase_index,:] =0 
		return dropout_data
