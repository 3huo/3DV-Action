import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from utils import group_points_2,group_points_2_3DV

nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
nstates_plus_3 = [256,512,1024,1024,256]

vlad_dim_out = 128*8



class PointNet_Plus(nn.Module):
    def __init__(self,opt,num_clusters=8,gost=1,dim=128,normalize_input=True):
        super(PointNet_Plus, self).__init__()
        self.temperal_num = opt.temperal_num
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM # x,y,x,c : 4
        self.num_outputs = opt.Num_Class


        self.dim=dim

        self.normalize_input=normalize_input

        self.pooling = opt.pooling
        #self._init_params()

        self.dim_out = 4096

        if self.pooling == 'concatenation':
        	self.dim_out = 1024*4
        # if self.pooling == 'bilinear':
        # 	self.dim_out = 4096

        self.net3DV_1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,64),stride=1)
            )

        self.net3DV_2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )

        self.net3DV_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )

        self.dim_drop1 = nn.Linear(nstates_plus_3[3], 64)
        self.dim_drop2 = nn.Linear(nstates_plus_3[3], 64)
        self.netR_FC = nn.Sequential(
            # B*1024
            #nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            #nn.BatchNorm1d(nstates_plus_3[3]),
            #nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(self.dim_out, nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )

        self.netR_C1 = nn.Sequential(
            # B*INPUT_FEATURE_NUM*sample_num_level1*knn_K
            nn.Conv2d(3, nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            )

        self.netR_C2 = nn.Sequential(
            # B*131*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )

        self.netR_C3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )


    def forward(self, xt, xs1, xs2, xs3, yt, ys1, ys2, ys3):

        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1


        ###----motion stream--------
        B,d,N,k = xt.shape
        #xt =xt.view(-1,d,N,k)
        xt = self.net3DV_1(xt)  
        xt = torch.cat((yt, xt),1).squeeze(-1)
        # B*(4+128)*sample_num_level1
        self.ball_radius2 = self.ball_radius2 + torch.randn(1)/120.0
        inputs_level2, inputs_level2_center = group_points_2_3DV(xt, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        # # B*131*sample_num_level2*knn_K
        xt = self.net3DV_2(inputs_level2)
        # # B*256*sample_num_level2*1
        # print('netR_2:',x1.shape,x1[0,:,4])
        xt = torch.cat((inputs_level2_center, xt),1)
        # # B*259*sample_num_level2*1
        xt = self.net3DV_3(xt).squeeze(-1).squeeze(-1)
       

        ###----apearance streams--------
        '''
        multiple streams shared one PointNet++
        '''
        xs = torch.cat((xs1.unsqueeze(1), xs2.unsqueeze(1), xs3.unsqueeze(1)),1)
        B,c,d,N,k = xs.shape
        xs =xs.view(-1,d,N,k)

        xs = self.netR_C1(xs)
        ys = torch.cat((ys1.unsqueeze(1),ys2.unsqueeze(1),ys3.unsqueeze(1)),1)

        ys = ys.view(-1,d,N,1)
        xs = torch.cat((ys, xs),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        inputs_level2_r, inputs_level2_center_r = group_points_2(xs, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        xs = self.netR_C2(inputs_level2_r)
        xs = torch.cat((inputs_level2_center_r, xs),1)
        xs = self.netR_C3(xs).squeeze(-1).squeeze(-1)
        xs = xs.view(B,-1)
        x = torch.cat((xt,xs),-1)

        
        #print(x.shape)
        # if self.pooling == 'bilinear':
        #     x1 = self.dim_drop1(x1)
        #     x2 = self.dim_drop2(x2)
        #     x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
        #     #print(x1.shape)
        #     x = x1*x2.unsqueeze(-1)
        #     x=x.view(-1,x1.size(-1)*x2.size(1))
        #     x = self.netR_FC(x)

        #print('x1:',x1.shape,'x2:',x2.shape)
        if self.pooling == 'concatenation':
            #x = torch.cat((x1, x2),1)
            #x = x.view(-1,self.dim_out)
            x = self.netR_FC(x)
            # B*num_label
        return x



