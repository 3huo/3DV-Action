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

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
#rose@ntu.edu.sg
sample_num_level1 = 512
sample_num_level2 = 128

TRAIN_IDS_60 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

TRAIN_IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,45,46,47,49,
			50,52,53,54,55,56,57,58,59,70,74,78,80,81,82,83,84,85,86,89,91,92,93,94,95,97,98,100,103]
TRAIN_VALID_IDS = ([1, 2, 5, 8, 9, 13, 14, 15, 16, 18, 19, 27, 28, 31, 34, 38], [4, 17, 25, 35])
compiled_regex = re.compile('.*S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3}).*')
SAMPLE_NUM = 2048

class NTU_RGBD(Dataset):
	"""NTU depth human masked datasets"""
	def __init__(self, root_path, opt,
		full_train = True,
		test=False, 
		validation=False,
		DATA_CROSS_VIEW = True,
		Transform = True):

		self.DATA_CROSS_VIEW = DATA_CROSS_VIEW
		self.root_path = root_path
		self.SAMPLE_NUM = opt.SAMPLE_NUM
		self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
		self.transform = Transform
		self.depth_path = opt.depth_path

		self.point_vids = os.listdir(self.root_path)#.sort()
		self.point_vids.sort()
		#print(self.point_vids)
		self.TRAIN_IDS = TRAIN_IDS
		if opt.dataset == 'ntu60':
			indx = self.point_vids.index('S017C003P020R002A060_xyzC.mat')#('S016C003P040R002A060_xyzC.mat')#('S017C003P020R002A060.npy')
			self.point_vids = self.point_vids[0:indx]
			self.TRAIN_IDS = TRAIN_IDS_60

		self.num_clouds = len(self.point_vids)
		print(self.num_clouds)
		self.point_data   = self.load_data()
	
		self.set_splits()

		self.id_to_action = list(pd.DataFrame(self.point_data)['action'] - 1)
		self.id_to_vidName = list(pd.DataFrame(self.point_data)['video_cloud_name'])

		self.train = (test == False) and (validation == False)
		if DATA_CROSS_VIEW == False:
			if test: self.vid_ids = self.test_split_subject.copy()
			elif validation: self.vid_ids = self.validation_split_subject.copy()
			elif full_train: self.vid_ids = self.train_split_subject.copy()
			else: self.vid_ids = self.train_split_subject_with_validation.copy()
		else:
			if test: self.vid_ids = self.test_split_camera.copy()
			else: self.vid_ids = self.train_split_camera.copy()

		print('num_data:',len(self.vid_ids))


		self.SAMPLE_NUM = opt.SAMPLE_NUM
		self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM

		self.point_clouds = np.empty(shape=[self.SAMPLE_NUM, self.INPUT_FEATURE_NUM],dtype=np.float32)

	def __getitem__(self, idx):
		vid_id = self.vid_ids[idx]
		vid_name = self.id_to_vidName[vid_id]
		S_idx = vid_name[1:4]
		#print(vid_name)
		v_name = vid_name[:-9]

		# 3 seg appearance data
		points_2048_f = np.zeros(shape = [3, self.SAMPLE_NUM, 3], dtype=np.float32)
		## appearance points load (3 seg)
		for s in range(3):
			v = int(np.random.rand()*5)+1 # random sample the points data in each segment from the saved points (here only save 5 frames point data)
			#appearance point data path
			path_v = 'NTU_3seg_depthpoint/Seg'+str(s+1)+'/NTU_v'+str(v)
			v_name1 = v_name+'_v'+str(v)+'.npy'
			path_points_frame = os.path.join(path_v,v_name1)
			points_2048 = np.load(path_points_frame) # load data

			## normalization for appearance point data
			min_p = np.amin(points_2048,axis = 0)
			max_p = np.amax(points_2048,axis = 0)
			len_xyz = max_p-min_p
			len_y = len_xyz[1]
			cent = (max_p+min_p)/2
			points_2048 = (points_2048-cent)/len_y
			points_2048_f[s] = points_2048

		## 3DV motion point data 
		path_cloud_npy = os.path.join(self.root_path,self.id_to_vidName[vid_id])
		# matlab data(.mat) OR python data(.npy)
		XYZ_C = sio.loadmat(path_cloud_npy)
		#print(self.id_to_vidName[vid_id])
		points_c = XYZ_C['pc'].astype(np.float32)
		#print(path_cloud_npy)
		#points_c= np.load(path_cloud_npy)
		points_c = np.expand_dims(points_c, axis=0)
		#print(points_c.shape, points_2048_f.shape)
		label = self.id_to_action[vid_id]

		# random angle rotate for data augment
		theta = np.random.rand()*1.4-0.7

		if self.transform:
			## point data augment
			points_c, points_2048_f = self.point_transform(points_c,points_2048_f,theta)

		points_2048_f = torch.tensor(points_2048_f,dtype=torch.float)
		label = torch.tensor(label)
		return points_c,points_2048_f[0],points_2048_f[1],points_2048_f[2],label,vid_name

	def __len__(self):
		return len(self.vid_ids)


	def load_data(self):
		self.point_data = []
		for cloud_idx in tqdm(range(self.num_clouds), "Getting video info"):
			self.point_data.append(self.get_pointdata(cloud_idx))

		return self.point_data

	def get_pointdata(self, vid_id):

		vid_name = self.point_vids[vid_id]
		match = re.match(compiled_regex, vid_name)
		setup, camera, performer, replication, action = [*map(int, match.groups())]
		return {
			'video_cloud_name': vid_name,
			'video_index': vid_id,
			'video_set':   (setup, camera),
			'setup':       setup,
			'camera':      camera,
			'performer':   performer,
			'replication': replication,
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
		train_ids_camera  = [2, 3]
		
		# Cross-Subject splits
		self.train_split_subject = list(
			dataset[dataset.performer.isin(self.TRAIN_IDS)]['video_index'])
		self.train_split_subject_with_validation = list(
			dataset[dataset.performer.isin(TRAIN_VALID_IDS[0])]['video_index'])
		self.validation_split_subject = list(
			dataset[dataset.performer.isin(TRAIN_VALID_IDS[1])]['video_index'])
		self.test_split_subject = list(
			dataset[~dataset.performer.isin(self.TRAIN_IDS)]['video_index'])

		# Cross-View splits
		self.train_split_camera  = list(
			dataset[dataset.camera.isin(train_ids_camera)]['video_index'])
		self.test_split_camera  = list(
			dataset[~dataset.camera.isin(train_ids_camera)]['video_index'])


	def point_transform(self,points_c,points_xyz,y):

		anglesX = (np.random.uniform()-0.5) * (1/9) * np.pi
		R_y = np.array([[[np.cos(y),0.0,np.sin(y)],
			[0.0,1.0,0.0],
			[-np.sin(y),0.0,np.cos(y)]]])
		R_x = np.array([[[1, 0, 0],
			[0, np.cos(anglesX), -np.sin(anglesX)],
 			[0, np.sin(anglesX), np.cos(anglesX)]]])
		#print(R_y.shape)

		points_c[:,:,0:3] = self.jitter_point_cloud(points_c[:,:,0:3],sigma=0.007, clip=0.04)
		points_xyz = self.jitter_point_cloud(points_xyz,sigma=0.012, clip=0.006)

		points_c[:,-1536:,:] = self.random_dropout_point_cloud(points_c[:,-1536:,:])
		points_xyz[:,-1536:,:] = self.random_dropout_point_cloud(points_xyz[:,-1536:,:])

		R =  np.matmul(R_y, R_x)
		
		points_c[:,:,0:3] = np.matmul(points_c[:,:,0:3],R)
		points_xyz = np.matmul(points_xyz,R)

		#if np.random.rand()>0.6:
		#	for i in range(3):
		#		points_c[:,i] = points_c[:,i]+(np.random.rand()-0.5)/6
		#		points_xyz[:,i] = points_xyz[:,i]+(np.random.rand()-0.5)/6
		
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

	def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
		"""

		:param data: Nx3 array
		:return: jittered_data: Nx3 array
		"""
		M,N, C = data.shape
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
