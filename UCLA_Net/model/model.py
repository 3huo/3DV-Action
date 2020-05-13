import torch
import math

import torch.nn as nn
import torch.nn.functional as F

from utils import group_points_2,group_points_2_pro

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

        self.alpha = 10
        self.num_clusters=num_clusters
        self.dim=dim
        self.gost=gost
        self.normalize_input=normalize_input
        self.fc=nn.Linear(dim,num_clusters+gost)
        self.centroids=nn.Parameter((torch.rand(dim,num_clusters+gost) * 2 - 1) * 0.2)
        self.pooling = opt.pooling
        #self._init_params()

        self.dim_out = 4096

        if self.pooling == 'concatenation':
                self.dim_out = 1024*4
        if self.pooling == 'bilinear':
                self.dim_out = 4096

        self.netDI_1 = nn.Sequential(
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

        self.netDI_2 = nn.Sequential(
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

        self.netDI_3 = nn.Sequential(
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
        self.netR_FC1 = nn.Sequential(
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

        # self.fc = nn.Sequential(
        #     encoding.nn.Normalize(),
        #     nn.Linear(64*64, 128),
        #     encoding.nn.Normalize(),
        #     nn.Linear(128, nclass))


    def _init_params(self):
        # nn.init.xavier_normal_(self.fc.weight.data)
        # nn.init.constant_(self.fc.bias.data,0.0)
        self.fc.weight = nn.Parameter(
            2.0 * self.alpha * self.centroids.transpose(1,0)
        )
        self.fc.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=0)*self.centroids.norm(dim=0)
        )

    def forward(self, xt, xs1, xs2, xs3, yt, ys1, ys2, ys3):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        B,d,N,k = xt.shape
        xt = self.netDI_1(xt)
        # B*128*sample_num_level1*1
        #print(xt.shape,yt.shape)
        xt = torch.cat((yt, xt),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        self.ball_radius2 = self.ball_radius2 + torch.randn(1)/120.0
        inputs_level2, inputs_level2_center = group_points_2_pro(xt, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1

        # B*131*sample_num_level2*knn_K
        xt = self.netDI_2(inputs_level2)
        # B*256*sample_num_level2*1
        xt = torch.cat((inputs_level2_center, xt),1)
        #print(xt.shape,inputs_level2_center.shape)
        # B*259*sample_num_level2*1

        xt = self.netDI_3(xt).squeeze(-1).squeeze(-1)


        xs = torch.cat((xs1.unsqueeze(1), xs2.unsqueeze(1), xs3.unsqueeze(1)),1)
        B,c,d,N,k = xs.shape
        xs =xs.view(-1,d,N,k)
        xs = self.netR_C1(xs)
        ys = torch.cat((ys1.unsqueeze(1),ys2.unsqueeze(1),ys3.unsqueeze(1)),1)
        ys = ys.view(-1,d,N,1)
        xs = torch.cat((ys, xs),1).squeeze(-1)
        inputs_level2_r, inputs_level2_center_r = group_points_2(xs, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        xs = self.netR_C2(inputs_level2_r)
        xs = torch.cat((inputs_level2_center_r, xs),1)
        #print(xs.shape)
        xs = self.netR_C3(xs).squeeze(-1).squeeze(-1)
        xs = xs.view(B,-1)
        x = torch.cat((xt,xs),-1)
        #print(x.shape)
        # B*1024
        if self.pooling == 'concatenation':
            #x = torch.cat((x1, x2),1)
            #x = x.view(-1,self.dim_out)
            #print(x.shape)
            x = self.netR_FC1(x)
            # B*num_label
        return x

'''
    def forward(self, x1, x2, y1, y2):
        #print(x.shape)
        #print(x.shape)
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*_*3*sample_num_level1*1

        # print('netR_0:',x1.shape,x1[0,:,4,:])
        x1 = self.netR_1(x1)  
        #print('netR_1:',x1.shape,x1[0,:,4,:])
        # B*128*sample_num_level1*1
        #print(x1.shape,y1.shape)
        x1 = torch.cat((y1, x1),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        self.ball_radius2 = self.ball_radius2 + torch.randn(1)/120.0
        inputs_level2, inputs_level2_center = group_points_2_pro(x1, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        # # B*131*sample_num_level2*knn_K
        x1 = self.netR_2(inputs_level2)
        # # B*256*sample_num_level2*1
        # print('netR_2:',x1.shape,x1[0,:,4])
        x1 = torch.cat((inputs_level2_center, x1),1)
        # # B*259*sample_num_level2*1
        x1 = self.netR_3(x1).squeeze(-1).squeeze(-1)
        # print('netR_3:',x1.shape,x1[0])
        #print('C0:',x2.shape,x2[0,:,4,:])
        x2 = self.netR_C1(x2)
        #print('C1:',x2.shape,x2[0,:,4,:])
        x2 = torch.cat((y2, x2),1).squeeze(-1)
        inputs_level2_r, inputs_level2_center_r = group_points_2(x2, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        x2 = self.netR_C2(inputs_level2_r)
        #print('C2:',x2.shape,x2[0,:,4])
        x2 = torch.cat((inputs_level2_center_r, x2),1)
        x2 = self.netR_C3(x2).squeeze(-1).squeeze(-1)
        #print('C3:',x2.shape,x2[0])

        
        if self.pooling == 'bilinear':
            x1 = self.dim_drop1(x1)
            x2 = self.dim_drop2(x2)
            x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
            #print(x1.shape)
            x = x1*x2.unsqueeze(-1)
            x=x.view(-1,x1.size(-1)*x2.size(1))
            x = self.netR_FC(x)

        #print('x1:',x1.shape,'x2:',x2.shape)
        if self.pooling == 'concatenation':
            x = torch.cat((x1, x2),1)
            #x = x.view(-1,self.dim_out)
            #print(x.shape)
            x = self.netR_FC(x)
            # B*num_label
        return x

        #def point_encoding_layer(self,xyz,features,hidden_size,npoint):

'''
class Attension_Point(nn.Module):
    def __init__(self,opt):
        super(Attension_Point, self).__init__()
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM # x,y,x,c : 4
        self.num_outputs = opt.Num_Class


        self.addnon = True

 
        self.netR_1 = nn.Sequential(
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
            nn.MaxPool2d((1,self.knn_K),stride=1)
            )

        self.netR_2 = nn.Sequential(
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

        self.netR_3 = nn.Sequential(
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
            # B*1024*1*1
        )

        self.max_3 = nn.Sequential(
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
        )

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

    def forward(self, xt, xs1, xs2, xs3, yt, ys1, ys2, ys3):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        B,d,N,k = xt.shape
        xt = self.netDI_1(x)
        # B*128*sample_num_level1*1
        xt = torch.cat((yt, xt),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        self.ball_radius2 = self.ball_radius2 + torch.randn(1)/120.0
        inputs_level2, inputs_level2_center = group_points_2(xt, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1

        # B*131*sample_num_level2*knn_K
        xt = self.netDI_2(inputs_level2)
        # B*256*sample_num_level2*1
        xt = torch.cat((inputs_level2_center, xt),1)
        # B*259*sample_num_level2*1

        xt = self.netDI_3(xt).squeeze(-1).squeeze(-1)


        xs = torch.cat((xs1.unsqueeze(1), xs2.unsqueeze(1), xs3.unsqueeze(1)),1)
        B,c,d,N,k = xs.shape
        xs =xs.view(-1,d,N,k)
        xs = self.netR_C1(xs)
        ys = torch.cat((ys1.unsqueeze(1),ys2.unsqueeze(1),ys3.unsqueeze(1)),1)
        ys = ys.view(-1,d,N,1)
        xs = torch.cat((ys, xs),1).squeeze(-1)
        inputs_level2_r, inputs_level2_center_r = group_points_2(xs, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        xs = self.netR_C2(inputs_level2_r)
        xs = torch.cat((inputs_level2_center_r, xs),1)
        xs = self.netR_C3(xs).squeeze(-1).squeeze(-1)
        xs = xs.view(B,-1)
        x = torch.cat((xt,xs),-1)
        #print(x.shape)
        # B*1024
        if self.pooling == 'concatenation':
            #x = torch.cat((x1, x2),1)
            #x = x.view(-1,self.dim_out)
            #print(x.shape)
            x = self.netR_FC(x)
            # B*num_label
        return x


    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal(self.g.weight)
        nn.init.constant(self.g.bias,0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal(self.W[0].weight)
            nn.init.constant(self.W[0].bias, 0)
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)

            
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in ['embedded_gaussian', 'dot_product']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            else:
                self.operation_function = self._dot_product

        elif mode == 'gaussian':
            self.operation_function = self._gaussian
        else:
            raise NotImplementedError('Mode concatenation has not been implemented.')

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

def nonlocalnet(input_layer,input_channel):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = NONLocalBlock2D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    else:
        net = NONLocalBlock2D(in_channels=input_channel,mode='embedded_gaussian')
        out = net(input_layer)
    return out
    

class PointNet_Plus_1(nn.Module):
    def __init__(self,opt):
        super(PointNet_Plus_1, self).__init__()
        self.knn_K = 80#opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM # x,y,x,c : 4
        self.num_outputs = opt.Num_Class
        
        self.netR_1 = nn.Sequential(
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
            nn.MaxPool2d((1,self.knn_K),stride=1)
            )

        self.netR_2 = nn.Sequential(
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
            nn.MaxPool2d((1,64),stride=1)
            # B*256*sample_num_level2*1
        )

        self.netR_3 = nn.Sequential(
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

        self.netR_FC = nn.Sequential(
            # B*1024
            #nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            #nn.BatchNorm1d(nstates_plus_3[3]),
            #nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(nstates_plus_3[3], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(512, self.num_outputs),
            # B*num_outputs
        )

    def forward(self, x,y):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        #print(x.shape)
        x = self.netR_1(x)
        # B*128*sample_num_level1*1
        y = y[:,:3,:,:]
       # print(x.shape,y.shape)
        x = torch.cat((y, x),1).squeeze(-1)
        # B*(3+128)*sample_num_level1
        self.ball_radius2 = self.ball_radius2-0.1 + torch.randn(1)/120.0
        inputs_level2, inputs_level2_center = group_points_2(x, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*131*sample_num_level2*knn_K, B*3*sample_num_level2*1
        #print(inputs_level2.shape)
        # B*131*sample_num_level2*knn_K
        x = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        x = torch.cat((inputs_level2_center, x),1)
        # B*259*sample_num_level2*1

        x = self.netR_3(x)
        # B*1024*1*1
        x = x.view(-1,nstates_plus_3[2])
        #print(x.shape)
        # B*1024
        x = self.netR_FC(x)
        # B*num_label
        return x
