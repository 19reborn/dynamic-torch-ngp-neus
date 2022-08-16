from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import get_encoder

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .embedder import get_embedder


class SDFNet(nn.Module):
    def __init__(self,
                 opt,
                 encoding="HashGrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 bound=1,
                 **kwargs
                 ):

        
        super().__init__()
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.bound = bound
        self.opt = opt

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoding = encoding

        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        # if encoder == None:
        #     if encoding == 'HashGrid':
        #         self.encoder = tcnn.Encoding(
        #             n_input_dims=3,
        #             encoding_config={
        #                 "otype": "HashGrid",
        #                 "n_levels": 16,
        #                 "n_features_per_level": 2,
        #                 "log2_hashmap_size": 19,
        #                 "base_resolution": 16,
        #                 "per_level_scale": per_level_scale,
        #             },
        #         )
        #         self.in_dim = 32

        #     elif encoding == 'Hash+PE':
        #         self.encoder_2 = tcnn.Encoding(
        #             n_input_dims=3,
        #             encoding_config={
        #                 "otype": "HashGrid",
        #                 "n_levels": 16,
        #                 "n_features_per_level": 2,
        #                 "log2_hashmap_size": 19,
        #                 "base_resolution": 16,
        #                 "per_level_scale": per_level_scale,
        #             },
        #         )
        #         self.in_dim = 32

        #         # self.multires = 6
        #         self.multires = 0
        #         self.encoder_1, input_ch = get_embedder(self.multires, input_dims=3)
        #         self.in_dim += input_ch
        #         # self.in_dim = input_ch
        #     else:
        #         self.multires = 6
        #         self.encoder, input_ch = get_embedder(self.multires, input_dims=3)
        #         self.in_dim = input_ch
        # else:
        #     self.encoder = encoder
        #     self.in_dim = 32


        sdf_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sdf + 15 SH features for color
            else:
                out_dim = hidden_dim
            # sdf_net.append(nn.Linear(in_dim, out_dim, bias=False)) # TO DO:: use tcnn network
            sdf_net.append(nn.Linear(in_dim, out_dim)) # TO DO:: use tcnn network
            if self.opt.geometry_init or self.opt.geometry_init_check:
                if l == num_layers - 1:
                    torch.nn.init.normal_(sdf_net[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(sdf_net[l].bias, -0.5)
                elif l == 0:
                    torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                    torch.nn.init.constant_(sdf_net[l].weight[:, 3:], 0.0)
                    torch.nn.init.normal_(sdf_net[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                else:
                    torch.nn.init.constant_(sdf_net[l].bias, 0.0)
                    torch.nn.init.normal_(sdf_net[l].weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

        self.sdf_net = nn.ModuleList(sdf_net)

        self.activation = nn.Softplus(beta=100)
        # self.activation = nn.ReLU()

    def forward(self, x):
        # x = self.encoder(x, bound=self.bound)
        # if self.encoding == 'HashGrid':
        #     x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        #     h = self.encoder(x)
        # elif self.encoding == 'Hash+PE':
        #     x_scaled = (x + self.bound) / (2 * self.bound) # to [0, 1]
        #     x_1 = self.encoder_1(x)
        #     x_2 = self.encoder_2(x_scaled)
        #     h = torch.cat([x_1,x_2],dim=-1)
            # h = torch.cat([x_1],dim=-1)

        x = self.encoder(x, bound=self.bound)
        h = x

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                # h = self.activation(h, inplace=True)
                h = self.activation(h)

        #sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        sdf = h[..., :1]
        geo_feat = h[..., 1:]
        
        return sdf, geo_feat

    def forward_with_gradients(self, x):
        print("not allowed!\n")
        exit(1)
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sdf, geo_feat = self.forward(x)
            gradients = torch.autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        # return sdf, gradients.detach(), geo_feat
        return sdf, gradients, geo_feat



class ColorNet(nn.Module):
    def __init__(self,
                 opt,
                 encoding_dir="SphericalHarmonics",
                 num_layers_color=3,
                 geo_feat_dim=15,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__()
        # color network
        self.opt = opt
        self.num_layers_color = num_layers_color        
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color


        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_dir = self.encoder_dir.n_output_dims
        
        self.in_dim_pos = 3
 
            
        self.in_dim_normal = 3
        self.bound = bound

        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_pos + self.in_dim_dir + self.in_dim_normal + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            # color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            color_net.append(nn.Linear(in_dim, out_dim))

        self.color_net = nn.ModuleList(color_net)


    def forward(self, points, normals, view_dirs, feature_vectors, mask=None):
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=points.dtype, device=points.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            points = points[mask]
            normals = normals[mask]
            view_dirs = view_dirs[mask]
            feature_vectors = feature_vectors[mask]
            
        view_dirs = (view_dirs + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        view_dirs = self.encoder_dir(view_dirs)
        # if self.encoder != None:
        #     points = (points + self.bound) / (2 * self.bound) # to [0, 1]
        #     points = self.encoder(points)           


        rendering_input = None

        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)


        h = rendering_input

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color= torch.sigmoid(h)
        return color

class SingleVarianceNet(nn.Module):
    def __init__(self, init_val = 0.3):
        super(SingleVarianceNet, self).__init__()
        # self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
        self.variance = nn.Parameter(torch.tensor(init_val))
        # self.variance = torch.tensor(init_val).cuda()

    def forward(self, x):
        return torch.ones([len(x), 1],device=self.variance.device) * torch.exp(self.variance * 10.0)
