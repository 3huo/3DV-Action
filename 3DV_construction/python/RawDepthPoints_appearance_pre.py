import os
import numpy as np
import random
import imageio


'''
To reduce the training time cost, the proposaled appearace raw depth are first converted into point data, 
then the points data can be send to PointNet++

For multi-appearance streams, each stream is to represent one temporal segment PoinNet++ 

Addtionally, in training process, the appearance frames should be random sampled from all video frames.
But to save storage space, only M frames in each segment are sampled to convert to points data. one of the M frames points is final send to PointNet++.
'''



save_path1 = '/NTU_3seg_depthpoint/'

root_path = 'ntu120dataset/'

seg_num=3 # 
M = 5 #
fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347
SAMPLE_NUM = 2048
sample_num_level1 = 512
sample_num_level2 = 128
def save_npy(data, save_path, filename):
	file = os.path.join(save_path, filename)
	if not os.path.isfile(file):
		np.save(file, data)

def farthest_point_sampling_fast(pc, sample_num):
	pc_num = pc.shape[0]
	sample_idx = np.zeros(shape = [sample_num,1], dtype = np.int32)
	sample_idx[0] = np.random.randint(0,pc_num)

	cur_sample = np.tile(pc[sample_idx[0],:], (pc_num,1))
	diff = pc-cur_sample
	min_dist = (diff*diff).sum(axis = 1)
	for cur_sample_idx in range(1,sample_num):
		sample_idx[cur_sample_idx] = np.argmax(min_dist)
		#print(cur_sample_idx,np.argmax(min_dist))
		if cur_sample_idx < sample_num-1:
			diff = pc - np.tile(pc[sample_idx[cur_sample_idx],:], (pc_num,1))
			#print(diff.shape,min_dist[valid_idx].shape)
			min_dist = np.concatenate((min_dist.reshape(pc_num,1), (diff*diff).sum(axis = 1).reshape(pc_num,1)), axis = 1).min(axis = 1)  ##?
	return sample_idx

def load_depth_from_img(depth_path):
	depth_im = imageio.imread(depth_path) #im is a numpy array
	return depth_im

def depth_to_pointcloud(depth_im):
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



sub_Files = os.listdir(root_path)
sub_Files.sort()

for s_fileName in sub_Files:

	videoPath = os.path.join(root_path, s_fileName, 'nturgb+d_depth_masked')
	if os.path.isdir(videoPath):
		print(s_fileName,'-------')
		video_Files = os.listdir(videoPath)
		#print(video_Files)
		video_Files.sort()
		for video_FileName in video_Files:
			print(video_FileName)
			filename = video_FileName +'_v3.npy'
			save_path = save_path1+'Seg1/NTU_v3'
			file = os.path.join(save_path,filename)
			#print(file,os.path.isfile(file))
			if os.path.isfile(file):
				print('ero')
				continue
			pngPath = os.path.join(videoPath,video_FileName)
			imgNames = os.listdir(pngPath)
			imgNames.sort()

			for ii in range(M):		
				for jj in range(seg_num):

					try:
						i = int(np.random.randint(int(len(imgNames)*jj/seg_num), int(len(imgNames)*(jj+1)/seg_num), size=1))
						path_raw_png = os.path.join(pngPath,imgNames[i])
						#print(path_raw_png)
						png =load_depth_from_img(path_raw_png)
					except IOError:
						print('error reading image!')
						i = int(np.random.randint(int(len(imgNames)*jj/seg_num), int(len(imgNames)*(jj+1)/seg_num), size=1))
						path_raw_png = os.path.join(pngPath,imgNames[i])
						print(path_raw_png)
						png =load_depth_from_img(path_raw_png)

					## to point  //  sample points
					raw_points_xyz = depth_to_pointcloud(png)
					## points sample 2048
					points_num = raw_points_xyz.shape[1]
					#print(raw_points_xyz.shape)
					all_sam = np.arange(points_num)
					if points_num<SAMPLE_NUM:
						if points_num < SAMPLE_NUM/2:
							index = random.sample(list(all_sam),SAMPLE_NUM-points_num-points_num)
							index.extend(list(all_sam))	
							index.extend(list(all_sam))	
						else:
							index = random.sample(list(all_sam),SAMPLE_NUM-points_num)
							index.extend(list(all_sam))
					else:
						index = random.sample(list(all_sam),SAMPLE_NUM)

					points_2048 = raw_points_xyz[:,index].T
					#print(points_2048.shape)
					## points sample 1st
					sampled_idx_l1 = farthest_point_sampling_fast(points_2048, sample_num_level1)
					other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1.ravel())
					new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
					points_2048 = points_2048[new_idx,:]
					## points sample 2nd
					sampled_idx_l2 = farthest_point_sampling_fast(points_2048[0:sample_num_level1,:], sample_num_level2)
					other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
					new_idx = np.concatenate((sampled_idx_l2.ravel(), other_idx))
					points_2048[0:sample_num_level1] = points_2048[new_idx,:]


					save_path1_ii = save_path1+'Seg'+str(jj+1)+'/NTU_v'+str(ii+1)
					filename = video_FileName +'_v'+str(ii+1)+'.npy'
					
					if not os.path.exists(save_path1_ii):
						print('new path:',save_path1_ii)
						os.makedirs(save_path1_ii)
					save_npy(points_2048,save_path1_ii,filename)




