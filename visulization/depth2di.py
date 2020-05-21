# -*- coding: utf-8 -*-
import os
import imageio
import numpy as np

from mayavi import mlab
file_dir = 'D:\\MY_Paper\\3ddi\\pointclouddemo\\S001C001P001R001A058'
#file_dir = 'F:\\error_samples\\S001C001P002R002A010_34'

fx = 365.481
fy = 365.481
cx = 257.346
cy = 210.347

voxel_size = 50

def load_depth_from_img(depth_path):
    depth_im = imageio.imread(depth_path) #im is a numpy array
    depth_im[0:2,:]=0
    depth_im[-1:-10,:]=0
    depth_im[:,0:2]=0
    depth_im[:,-1:-10]=0
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

each_frame_points_num = np.zeros(n_frame,dtype = np.int32)
all_frame_points_list = [] 
i = 0
for png in imgNames[0:n_frame]:
    print(png)
    img_path = os.path.join(file_dir,png)
    depth_im = load_depth_from_img(img_path)
    cloud_im = depth_to_pointcloud(depth_im)
    all_frame_points_list.append(cloud_im)
    each_frame_points_num[i] = cloud_im.shape[1]
    i = i+1
     
all_frame_points_array = np.zeros(shape = (3,each_frame_points_num.sum()))
n_frame = len(each_frame_points_num)

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

#voxel_frames = np.zeros(shape = [n_frame, dx+1, dy+1, dz+1])


#n_frame
for t in range(1):
#for t in range(1):
    
    start_frame = 5
    if start_frame <0:
        start_frame = 0

    end_frame = 30#30#n_frame-20
    
    print('start:',start_frame,'end:',end_frame)
    voxel_DI = np.zeros(shape =[dx+1, dy+1, dz+1], dtype=np.float32)
    for i_frame in range(n_frame):
        threeD_matrix = np.zeros(shape=[dx+1,dy+1,dz+1], dtype=np.float32)
        for itm in all_frame_points_list[i_frame].T:
            #print(int((itm[0]-min_x)/voxel_size),int((itm[1]-min_y)/voxel_size),int((itm[2]-min_z)/voxel_size))
            #print(min_x,max_x)
           # if itm[0]<min_x+600 or itm[0]>max_x-200:
                #continue
            threeD_matrix[int((itm[0]-min_x)/voxel_size)][int((itm[1]-min_y)/voxel_size)][int((itm[2]-min_z)/voxel_size)] = 1
#            if i_frame ==0:
#            xx, yy, zz = np.where(threeD_matrix[:,:,:] != 0)
#            cc = voxel_DI[xx,yy,zz]
#            #cc = (cc-cc.min())/(cc.max()-cc.min())
#
#            nodes = mayavi.mlab.points3d(xx, yy, zz,colormap="hot", mode="sphere", scale_factor="0.05") #autumn mode="cube" "sphere"
#            #nodes = mayavi.mlab.points3d(xx, yy, zz ,mode="cube", scale_factor="1")
#            nodes.glyph.scale_mode = 'scale_by_vector'
#            #nodes.mlab_source.dataset.point_data.scalars = labels
#            mayavi.mlab.show()
#

    #voxel_frames[i_frame,:,:,:] = threeD_matrix
    #print(np.where(threeD_matrix!=0))
        if i_frame<end_frame and i_frame>=start_frame:
            idx_f = i_frame - start_frame
            len_f = end_frame - start_frame
            voxel_DI = voxel_DI + (idx_f*2-len_f + 1)*threeD_matrix
            frame_matrix = threeD_matrix

    
    #xx, yy, zz = np.where(voxel_DI[:,:,:] != 0)
    xx, yy, zz = np.where(voxel_DI[:,:,:] != 0)
    cc = voxel_DI[xx,yy,zz]


    cen = (cc.max()+cc.min())/2
    cc = 2*(cc-cen)/(cc.max()-cc.min())
    #cc = np.zeros(len(xx))
    ###
    '''
space = [1,35,0,38]#[15,28,10,27] #[1,40,0,6]#

func_x = lambda d: d>=space[0] and d<space[1]
func_y = lambda d: d>=space[2] and d<space[3]

c_x = np.vectorize(func_x)(xx)
c_y = np.vectorize(func_y)(yy)
c = c_x*c_y
index = np.where(c)
#cc[index] = 1
#v_selected = cc[index]
#np.std(v_selected)
xx, yy, zz, cc = xx[index], yy[index], zz[index], cc[index]
#
    '''

    scale = 0.9#0.83
    cc[cc>0] = np.power(cc[cc>0],scale)
    cc[cc<0] = -np.power(-cc[cc<0],scale)

    #xx, yy, zz = all_frame_points_list[12]

    #mlab.view(azimuth= 360*t/duration, distance=9)


    mlab.figure(size=(800, 800),fgcolor=(0, 0, 0), bgcolor=(1,1,1))
    zzz =zz-0.5
    #nodes = mlab.points3d(xx, yy, zzz, mode="cylinder", scale_factor="0.8",color = (0.3,0.3,0),resolution=100) 
    #nodes = mlab.points3d(-xx, -yy, zzz, mode="2dsquare", scale_factor="1",color = (0,0,0),line_width=1,resolution=100) 

    #nodes = mlab.points3d(-xx, -yy, zz, mode="cube", scale_factor="1",resolution=100) 
    nodes = mlab.points3d(-xx, -yy, zz, cc ,colormap="Blues", mode="cube", scale_factor="0.8",resolution=800) #autumn mode="cube" "sphere"
    ##nodes = mlab.points3d(xx, yy, zz ,mode="cube", scale_factor="1")

    nodes.glyph.scale_mode = 'scale_by_vector'
    
    #mayavi.mlab.axes(xlabel='x', ylabel='y', zlabel='z')#extent=
    #mlab.outline(nodes,color=(1, 0, 0),line_width=2)
    #nodes.mlab_source.dataset.point_data.scalars = labels
    #mlab.world_to_display()
    #mlab.pitch(60)
    #mlab.yaw(60)
    #mlab.view(azimuth= 92, elevation=80,distance=150,roll = 80)
    #mlab.view(azimuth= '92', elevation=60,distance=150,roll = 66)
    print(t)
    #t=13
    #mlab.view(azimuth= t,distance=180,elevation=88)
    #
    #name = 'v_voxels_png/frame_' + str(t) + '.png'
#    print(name)
   # mlab.view(azimuth= 00, elevation=185,distance=140,roll = None)
    mlab.show()
    #mlab.savefig(filename =name)
   # mlab.close()
    #mlab.colorbar()
    #for t in range(10):
    #    mlab.view(azimuth= 360*t/2, distance=9)
    #    name = '/v_png/frame_' + str(t) + '.png'
    #    mlab.savefig(filename =name)
    
    #Accent' or 'Blues' or 'BrBG' or 'BuGn' or 'BuPu' or 'CMRmap' or 
    #'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 
    #'PRGn' or 'Paired' or 'Pastel1' or 'Pastel2' or 'PiYG' or 'PuBu' or    
    #'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' 
    #or 'RdYlBu' or 'RdYlGn' or 'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' 
    #or 'Vega10' or 'Vega20' or 'Vega20b' or 'Vega20c' or 'Wistia' or 'YlGn' or 
    #'YlGnBu' or 'YlOrBr' or 'YlOrRd' or 'afmhot' or 'autumn' or 'binary' or 
    #'black-white' or 'blue-red' or 'bone' or 'brg' or 'bwr' or 'cool' or 'coolwarm' 
    #   or 'copper' or 'cubehelix' or 'file' or 'flag' or 'gist_earth' or 'gist_gray' 
    #or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or 'gist_yarg'
    # or 'gnuplot' or 'gnuplot2' or 'gray' or 'hot' or 'hsv' or 'inferno' or 'jet'
    # or 'magma' or 'nipy_spectral' or 'ocean' or 'pink' or 'plasma' or 'prism' 
    # or 'rainbow' or 'seismic' or 
    #'spectral' or 'spring' or 'summer' or 'terrain' or 'viridis' or 'winter'




