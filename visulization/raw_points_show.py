# -*- coding: utf-8 -*-
import os 
import imageio
import numpy as np
import mayavi.mlab

#file_dir = 'D:\\MY_Paper\\3ddi\\pointclouddemo\\S001C001P001R001A058'
file_dir = 'D:\\MY_Paper\\3ddi\\pointclouddemo\\S026C002P076R002A098'
file_dir = 'D:\\MY_Paper\\3ddi\\data_depth\\Make_OK_sign\\S025C003P068R001A071'
file_dir = 'D:\\MY_Paper\\3ddi\\pointclouddemo\\S001C001P001R001A051_ljy\\S001C001P001R001A051'
#file_dir = 'D:\\MY_Paper\\3ddi\\data_depth\\Move_object\\S019C003P051R002A092'
fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347

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





imgNames = os.listdir(file_dir)
imgNames.sort()
n_frame = len(imgNames)

for t in range(1):
#for t in range(22,n_frame):
    i_f = 1
    png = imgNames[i_f]
    print(png)
    img_path = os.path.join(file_dir,png)
    depth_im = load_depth_from_img(img_path)
    cloud_im = depth_to_pointcloud(depth_im)

    xx= cloud_im[0,:]
    yy= cloud_im[1,:]
    zz= cloud_im[2,:]


    mayavi.mlab.figure(fgcolor=(0.5, 0.5, 0.5), bgcolor=(1, 1, 1))
    nodes = mayavi.mlab.points3d(-xx, -yy, zz ,mode="cube", scale_factor="10")
    nodes.glyph.scale_mode = 'scale_by_vector'
    #mayavi.mlab.colorbar()
    #nodes.mlab_source.dataset.point_data.scalars = labels
    mayavi.mlab.view(azimuth= 00, elevation=185,distance=9800,roll = None)
    name = 'v_points_lujy_png/frame_' + str(t) + '.png'
    #mayavi.mlab.savefig(filename =name)
    #mayavi.mlab.close()
    mayavi.mlab.show()
