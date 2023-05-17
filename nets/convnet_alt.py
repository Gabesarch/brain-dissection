import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

# import ipdb
# st = ipdb.set_trace

class ConvNet(nn.Module):
    def __init__(self, num_voxels, W, H, hid_channels=48, kernel_size=3, useneuralpredictors=False):
        super(ConvNet, self).__init__()

        self.num_voxels = num_voxels
        self.W, self.H = W, H
        self.useneuralpredictors = useneuralpredictors

        padding = int((kernel_size-1)/2)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hid_channels, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace = True)
        )

        self.conv_block1 = ConvBlock(hid_channels=hid_channels, kernel_size=5)
        self.conv_block2 = ConvBlock(hid_channels=hid_channels, kernel_size=kernel_size)
        self.conv_block3 = ConvBlock(hid_channels=hid_channels, kernel_size=kernel_size)
        self.conv_block4 = ConvBlock(hid_channels=hid_channels, kernel_size=kernel_size, pool=False)
        
        if useneuralpredictors:
            from neuralpredictors.neuralpredictors.layers.readouts import SpatialXFeatureLinear, FullGaussian2d
            self.readout = SpatialXFeatureLinear(torch.tensor((hid_channels, int(W/(2**3)), int(H/(2**3)))), num_voxels,  bias = True) 
        else:
            # spatial and feature weights
            self.W_spatial = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty((int((W/(2**3))*(H/(2**3))), num_voxels)), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True) #nn.Conv2d(in_channels=hid_channels, out_channels=12, kernel_size=1, stride=1)
            self.W_features = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty((hid_channels, num_voxels)), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True)
            # self.W_features = torch.nn.Parameter(nn.init.trunc_normal_(torch.empty(num_voxels), mean=0, std=0.01, a=-0.01*2, b=0.01*2), requires_grad=True)
            # bias term for prediction
            self.b_out = torch.nn.Parameter(nn.init.constant_(torch.empty(num_voxels), 0), requires_grad=True)

        self.dropout = nn.Dropout(0.1)

    def get_last_activation_sizes(self):
        return int(self.W/(2**3)), int(self.H/(2**3))

    def forward(self, x):
        """Forward pass.
        Args:
            x (PIL): input data (image)
        Returns:
            tensor: depth
        """

        b = x.shape[0]

        x = self.initial_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        ####### pre-compute outer product version ######
        # torch.bmm(self.W_spatial.T.unsqueeze(2), self.W_features.cuda().T.unsqueeze(1))
        ###############################################

        if self.useneuralpredictors:
            out = self.readout(x)
        else:
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

            # # the above is computationally the same as doing the following: 
            # x_flatten_spatial2 = x_flatten_spatial.unsqueeze(-1).expand(b,1,784,48,649)
            # W_spatial2 = W_spatial.unsqueeze(0).permute(0,2,3,4,1).expand(b,1,784,48,649)
            # W_features2 = W_features.unsqueeze(1).unsqueeze(1).expand(b,1,784,48,649)
            # h_out2 = x_flatten_spatial2 * W_spatial2 * W_features2
            # h_out2 = h_out2.sum((1,2,3))

            out = h_out + self.b_out.unsqueeze(0).expand(b, self.num_voxels)

        return out

    def get_sparsity_loss(self):
        if self.useneuralpredictors:
            l1_spatial = torch.mean(torch.norm(self.readout.spatial, 1, dim=0)) #torch.sum(torch.abs(self.W_spatial), dim=0)
            # feature_params = self.W_features.view(-1) #torch.cat([x.view(-1) for x in self.W_features.parameters()])
            l1_feature = torch.mean(torch.norm(self.readout.features, 1, dim=0)) #torch.norm(feature_params, 1) #torch.sum(torch.abs(self.W_features), dim=0)
        else:
            # number of units can change so we take norm across mask and mean across units
            # spatial_params = self.W_spatial #torch.cat([x.view(-1) for x in self.initial_conv.parameters()])
            l1_spatial = torch.mean(torch.norm(self.W_spatial, 1, dim=0)) #torch.sum(torch.abs(self.W_spatial), dim=0)
            # feature_params = self.W_features.view(-1) #torch.cat([x.view(-1) for x in self.W_features.parameters()])
            l1_feature = torch.mean(torch.norm(self.W_features, 1, dim=0)) #torch.norm(feature_params, 1) #torch.sum(torch.abs(self.W_features), dim=0)
            #     l1_reg = torch.sum(l1_spatial * l1_feature)
            # else:
            #     l1_reg = torch.sum(l1_spatial)
        return l1_spatial, l1_feature

    def get_spatial_mask(self):
        if self.useneuralpredictors:
            return self.readout.spatial.reshape(self.readout.spatial.shape[0], -1).permute(1,0)
        else:
            return self.W_spatial

    def get_voxel_feature_maps(self, x, get_voxel_response=False, use_spatial_mask=False):

        b = x.shape[0]

        x = self.initial_conv(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        if self.useneuralpredictors:
            features = self.readout.features.transpose(0,1)
            bias = self.readout.bias.view(1, -1, 1, 1) 
            voxel_maps = torch.einsum('bcwh,cn->bnwh', x, features) + bias
            if get_voxel_response:
                response = self.readout(x)
            return voxel_maps, response
        else:
            feature_filter = self.W_features.permute(1,0).unsqueeze(-1).unsqueeze(-1)
            if use_spatial_mask:
                W_spatial = torch.abs(self.W_spatial)
                # normalized independently for each voxel by dividing each 
                # spatial weight by the square-root of the sum of squared 
                # spatial weights across all locations
                # W_spatial_norm = torch.sqrt(torch.sum(W_spatial**2, 0))
                W_spatial_norm = torch.linalg.norm(W_spatial, ord=2, dim=0)
                W_spatial = W_spatial / W_spatial_norm.unsqueeze(0).expand(W_spatial.shape[0], self.num_voxels)
                W_spatial = W_spatial.permute(1,0).unsqueeze(1)
                voxel_maps = feature_filter.squeeze(-1) * W_spatial
                voxel_maps = voxel_maps.sum(1) # sum over channels
                voxel_maps = voxel_maps + self.b_out.view(1, -1, 1, 1) 
            else:
                voxel_maps = F.conv2d(x, feature_filter) + self.b_out.view(1, -1, 1, 1) 

            if get_voxel_response:
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
                h_out_sum = h_out.sum(1) # mean of channels
                response = h_out_sum + self.b_out.unsqueeze(0).expand(b, self.num_voxels)
                return voxel_maps, response

        return voxel_maps

class ConvBlock(nn.Module):
    def __init__(self, hid_channels=48, kernel_size=3, stride=1, pool=True):
        super(ConvBlock, self).__init__()

        padding = int((kernel_size-1)/2)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace = True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace = True)
        )
        if pool:
            self.pool = nn.AvgPool2d(kernel_size  = 2, stride = 2)
        else:
            self.pool = nn.AvgPool2d(kernel_size  = 2, stride = 1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.pool is not None:
            x = self.pool(x)
        return x