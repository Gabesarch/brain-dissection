import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from e2cnn import gspaces
from e2cnn import nn as e2nn
from arguments import args

'''
For using this model, download https://github.com/QUVA-Lab/e2cnn
'''

# import ipdb
# st = ipdb.set_trace

class ConvNet(nn.Module):
    def __init__(self, num_voxels, W, H, hid_channels=48, kernel_size=3):
        super(ConvNet, self).__init__()

        self.num_voxels = num_voxels

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # we choose 48 feature fields, each transforming under the regular representation of C8
        out_type = e2nn.FieldType(self.r2_act, hid_channels*[self.r2_act.regular_repr])
        self.initial_conv = e2nn.SequentialModule(
            # nn.MaskModule(in_type, 29, margin=1),
            e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=int((kernel_size-1)/2)),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )

        self.conv_block1 = ConvBlockE2Steerable(r2_act=self.r2_act, hid_channels=hid_channels, kernel_size=kernel_size)
        self.conv_block2 = ConvBlockE2Steerable(r2_act=self.r2_act, hid_channels=hid_channels, kernel_size=kernel_size)
        self.conv_block3 = ConvBlockE2Steerable(r2_act=self.r2_act, hid_channels=hid_channels, kernel_size=kernel_size)
        self.conv_block4 = ConvBlockE2Steerable(r2_act=self.r2_act, hid_channels=hid_channels, kernel_size=kernel_size)

        self.gpool = e2nn.GroupPooling(out_type)
        
        # spatial and feature weights
        self.W_spatial = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty((int((W/(2**4))*(H/(2**4))), num_voxels)), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True) #nn.Conv2d(in_channels=hid_channels, out_channels=12, kernel_size=1, stride=1)
        self.W_features = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty((hid_channels, num_voxels)), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True)
        # self.W_features = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty(num_voxels), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True)

        # bias term for prediction
        self.b_out = torch.nn.Parameter(nn.init.constant_(torch.empty(num_voxels), 0), requires_grad=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (PIL): input data (image)
        Returns:
            tensor: depth
        """

        b = x.shape[0]

        # run vit
        x = e2nn.GeometricTensor(x, self.input_type)

        x = self.initial_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.gpool(x)

        x = x.tensor

        ####### pre-compute outer product version ######
        # torch.bmm(self.W_spatial.T.unsqueeze(2), self.W_features.cuda().T.unsqueeze(1))
        ###############################################
        
        # 1x1 convolution for spatial filter (for each voxel)
        x_flatten_spatial = x.flatten(2,3).permute(0,2,1).unsqueeze(1)
        W_spatial = torch.abs(self.W_spatial)
        # normalized independently for each voxel by dividing each 
        # spatial weight by the square-root of the sum of squared 
        # spatial weights across all locations
        # W_spatial_norm = torch.sqrt(torch.sum(W_spatial**2, 0))
        W_spatial_norm = torch.linalg.norm(W_spatial, ord=2, dim=0)
        W_spatial = W_spatial / W_spatial_norm.unsqueeze(0).expand(W_spatial.shape[0], self.num_voxels)
        W_spatial = W_spatial.permute(1,0).unsqueeze(1).unsqueeze(-1)
        h_spatial = F.conv2d(x_flatten_spatial, W_spatial, bias=None)
        h_spatial = h_spatial.squeeze(2).permute(0,2,1)

        # feature filter (for each voxel)
        W_features = self.W_features.unsqueeze(0).expand(b, h_spatial.shape[1], h_spatial.shape[2])
        # W_features = torch.abs(W_features)
        h_out = W_features * h_spatial

        # get prediction with bias
        h_out = h_out.sum(1) # mean of channels
        out = h_out + self.b_out.unsqueeze(0).expand(b, self.num_voxels)

        return out

    def get_sparsity_loss(self):
        l1_spatial = torch.sum(torch.abs(self.W_spatial), dim=0)
        if args.sparsity_on_feature_weights:
            l1_feature = torch.sum(torch.abs(self.W_features), dim=0)
            l1_reg = torch.sum(l1_spatial * l1_feature)
        else:
            l1_reg = torch.sum(l1_spatial)
        return l1_reg

    def get_voxel_feature_maps(self, x):

        b = x.shape[0]

        # run vit
        x = e2nn.GeometricTensor(x, self.input_type)

        x = self.initial_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.gpool(x)

        x = x.tensor

        feature_filter = self.W_features.permute(1,0).unsqueeze(-1).unsqueeze(-1)

        voxel_maps = F.conv2d(x, feature_filter)

        return voxel_maps





class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.blurpool = antialiased_cnns.BlurPool(12, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.blurpool(x)
        return x

class ConvBlockE2Steerable(nn.Module):
    def __init__(self, r2_act, hid_channels=48, kernel_size=3):
        super(ConvBlockE2Steerable, self).__init__()

        padding = int((kernel_size-1)/2)

        # the output type of the second convolution layer are 48 regular feature fields of C8
        in_type = e2nn.FieldType(r2_act, hid_channels*[r2_act.regular_repr])
        out_type = in_type
        self.block1 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = e2nn.SequentialModule(
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        return x