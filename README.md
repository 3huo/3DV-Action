# 3DV: 3D Dynamic Voxel for Action recognition in depth video.
> Paper [pdf](http://arxiv.org/abs/2005.05501)

To be represented in CVPR2020

### Usege:
Our model consists of 2 parts.
The first part is about 3DV point data generation and apprearance point data preparation.
For example, "3DV_construction\python\ntu120_3dv_pre.py" is used to generate the 3DV point data with multiple motion features. The output 3DV points data form is N*(x,y,z,m_g,m_1,...,m_T)
"3DV_construction\python\RawDepthPoints_appearance_pre.py" is used to sample multi-segment temporal frames with its raw points into the form of  N*(x,y,z).

The second part is about multi-stream PointNet++. The generated point data in part 1 can be fed to Network. 



Some related data:

https://pan.baidu.com/s/1TtApXBQx4sNi4ZmTOL6lSw code:k0jt