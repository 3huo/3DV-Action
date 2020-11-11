import os 
import tqdm
import imageio
import numpy as np
import time
import random


'''
due to the ntu120 full depth maps data is not avialable, the action proposal is used by skeleton-based action proposal.  
'''


fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347

voxel_size = 35 
M =5 #temporal splits


SAMPLE_NUM = 2048
sample_num_level1 = 512
sample_num_level2 = 128
K = 60  # max frame limit for temporal rank
save_path = '/3DV_pointdata/NTU_voxelsize35_split5'

try:
	os.makedirs(save_path)
except OSError:
	pass

def main():
	data_path = '/ntu120dataset'
	sub_Files = os.listdir(data_path)
	sub_Files.sort()

	for s_fileName in sub_Files:

		videoPath = os.path.join(data_path, s_fileName, 'nturgb+d_depth_masked')
		if os.path.isdir(videoPath):
			print(s_fileName)
			video_Files = os.listdir(videoPath)
			#print(video_Files)
			video_Files.sort()

			for video_FileName in video_Files:
				print(video_FileName)
				filename = video_FileName +'.npy'
				file = os.path.join(save_path, filename)
				if os.path.isfile(file):
					continue

				pngPath = os.path.join(videoPath,video_FileName)
				imgNames = os.listdir(pngPath)
				imgNames.sort()
				#print(imgNames)

				## ------ select a fixed number K of images
				n_frame = len(imgNames)
				all_sam = np.arange(n_frame)

				if n_frame	> K:
					frame_index = random.sample(list(all_sam),K)
					#frame_index = np.array(frame_index)
					n_frame = K
				else:	
					frame_index = all_sam.tolist()

				frame_index.sort()		

				### ------convert the depth sequence to points data
				each_frame_points_num = np.zeros(n_frame,dtype = np.int32)
				all_frame_points_list = []

				i = 0
				for i_frame in frame_index:
					depthName = imgNames[i_frame]
					img_path = os.path.join(pngPath,depthName)
					depth_im = load_depth_from_img(img_path)
					cloud_im = depth_to_pointcloud(depth_im)
					all_frame_points_list.append(cloud_im) #all frame points in 1 list
					each_frame_points_num[i] = cloud_im.shape[1]
					i = i+1

				all_frame_points_array = np.zeros(shape = (3,each_frame_points_num.sum()))

				## compute the Max bounding box to voxelization
				for i_frame in range(n_frame):
					start_idx = each_frame_points_num[0:i_frame].sum()
					all_frame_points_array[:, start_idx:start_idx + each_frame_points_num[i_frame]] = all_frame_points_list[i_frame]

				max_x = all_frame_points_array[0,:].max()
				max_y = all_frame_points_array[1,:].max()
				max_z = all_frame_points_array[2,:].max()
				min_x = all_frame_points_array[0,:].min()
				min_y = all_frame_points_array[1,:].min()
				min_z = all_frame_points_array[2,:].min()

				dx, dy, dz = map(int, [(max_x-min_x)/voxel_size, (max_y-min_y)/voxel_size, (max_z-min_z)/voxel_size])	
				voxel_DI = np.zeros(shape =[M, dx+1, dy+1, dz+1])
				#avg_threeD_matrix = np.zeros(shape=[dx+1,dy+1,dz+1], dtype=np.float32)
				for i_frame in range(n_frame):
					threeD_matrix = np.zeros(shape=[dx+1,dy+1,dz+1], dtype=np.float32)
					## voxelization
					for itm in all_frame_points_list[i_frame].T:
						threeD_matrix[int((itm[0]-min_x)/voxel_size)][int((itm[1]-min_y)/voxel_size)][int((itm[2]-min_z)/voxel_size)] = 1
					
					# 3dv generate:
					'''
					fast version rank pooling: sum_i (i*2-N-1)*V_i
					
					'''
					# #V_i version1：average voxel from 1 to i
					# threeD_matrix = (threeD_matrix+avg_threeD_matrix)/(i_frame+1)

					# #V_i version2：ith voxel
					for m in range(M):
						## first segment 3dv construction: all frame voxel is used!   motion:(m_g) 
						if m == 0:
							voxel_DI[0,:,:,:] = voxel_DI[0,:,:,:] + (i_frame*2-n_frame+1)*threeD_matrix
						## T_1=M-1 segments 3dv construction: M-1 temporal splits with 0.5 overlap
						if m == 1 and i_frame<round(n_frame*2/5):
							idx_f = i_frame
							len_f = round(n_frame*2/5)
							voxel_DI[m,:,:,:] = voxel_DI[m,:,:,:] + (idx_f*2-len_f+1)*threeD_matrix
						if m == 2 and i_frame<round(n_frame*3/5) and i_frame>=round(n_frame*1/5):
							idx_f = i_frame - round(n_frame*1/5)
							len_f = round(n_frame*3/5) - round(n_frame*1/5)
							voxel_DI[m,:,:,:] = voxel_DI[m,:,:,:] + (idx_f*2-len_f+1)*threeD_matrix
						if m == 3 and i_frame<round(n_frame*4/5) and i_frame>=round(n_frame*2/5):
							idx_f = i_frame - round(n_frame*2/5)
							len_f = round(n_frame*4/5) - round(n_frame*2/5)
							voxel_DI[m,:,:,:] = voxel_DI[m,:,:,:] + (idx_f*2-len_f+1)*threeD_matrix
						if m ==4 and i_frame>=round(n_frame*3/5):
							idx_f = i_frame - round(n_frame*3/5)
							len_f = n_frame - round(n_frame*3/5)
							voxel_DI[m,:,:,:] = voxel_DI[m,:,:,:] + (idx_f*2-len_f+1)*threeD_matrix


				### 3DV voxel to 3DV points
				mm, xx, yy, zz = np.where(voxel_DI[:,:,:,:] != 0) 
				xyz = np.column_stack((xx,yy,zz))
				xyz = np.unique(xyz, axis = 0)
				motion = voxel_DI[:, xyz[:,0], xyz[:,1], xyz[:,2]]
				points_xyzc = np.concatenate((xyz,motion.T), axis = 1)  #final 3DV point feature shape N*(x,y,z,m_g,m_1,...m_t)

				### Sample and normalization
				if len(xx)< SAMPLE_NUM:
					rand_points_index = np.random.randint(0, points_xyzc.shape[0], size=SAMPLE_NUM-len(xx))
					points_xyzc = np.concatenate((points_xyzc, points_xyzc[rand_points_index,:]), axis = 0)
				else:
					rand_points_index = np.random.randint(0, points_xyzc.shape[0], size=SAMPLE_NUM)
					points_xyzc = points_xyzc[rand_points_index,:]
				## Normalization
				y_len = points_xyzc[:,1].max() - points_xyzc[:,1].min()
				x_len = points_xyzc[:,0].max() - points_xyzc[:,0].min()
				z_len = points_xyzc[:,2].max() - points_xyzc[:,2].min()
				c_max, c_min = points_xyzc[:,3:8].max(axis=0), points_xyzc[:,3:8].min(axis=0)
				c_len = c_max - c_min

				x_center = (points_xyzc[:,0].max() + points_xyzc[:,0].min())/2
				y_center = (points_xyzc[:,1].max() + points_xyzc[:,1].min())/2
				z_center = (points_xyzc[:,2].max() + points_xyzc[:,2].min())/2
				centers = np.tile([x_center,y_center,z_center],(SAMPLE_NUM,1))
				points_xyzc[:,0:3] = (points_xyzc[:,0:3]-centers)/y_len
				points_xyzc[:,3:8] = (points_xyzc[:,3:8] - c_min)/c_len - 0.5

				## FPS Sampling for PointNet++
				# lst
				sampled_idx_l1 = farthest_point_sampling_fast(points_xyzc[:,0:3], sample_num_level1)
				other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1.ravel())
				new_idx = np.concatenate((sampled_idx_l1.ravel(), other_idx))
				points_xyzc = points_xyzc[new_idx,:]
					
				# 2nd
				sampled_idx_l2 = farthest_point_sampling_fast(points_xyzc[0:sample_num_level1,0:3], sample_num_level2)
				other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
				new_idx = np.concatenate((sampled_idx_l2.ravel(), other_idx))
				points_xyzc[0:sample_num_level1] = points_xyzc[new_idx,:]
				# if points_xyzc.shape[0] != SAMPLE_NUM:
				# 	print('error sample!',len(sampled_idx_l2),len(sampled_idx_l1))
				# 	points_xyzc = points_xyzc[0:SAMPLE_NUM,:]
				save_npy(points_xyzc, filename)



def save_npy(data, filename):
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
		## find the farthest point

		sample_idx[cur_sample_idx] = np.argmax(min_dist)
		if cur_sample_idx < sample_num-1:
			diff = pc - np.tile(pc[sample_idx[cur_sample_idx],:], (pc_num,1))
			min_dist = np.concatenate((min_dist.reshape(pc_num,1), (diff*diff).sum(axis = 1).reshape(pc_num,1)), axis = 1).min(axis = 1)  ##?
	#print(min_dist)
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

if __name__ == '__main__':
	main()
