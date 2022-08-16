import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from encoding import get_encoder
import tinycudann as tcnn
from activation import trunc_exp
from .neus_renderer import NeuSRenderer
from .neus_base_tcnn import SDFNet, ColorNet, SingleVarianceNet
# from .original_neus_network_base import SDFNetwork, RenderingNetwork

class NeusNetwork(NeuSRenderer):
    def __init__(self,
                 opt,
                 encoding="tiledgrid",
                #  encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency", # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_deform=5, # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(opt, bound, **kwargs)
        
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
        self.opt = opt

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)


        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time # grid dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 3 # deformation for xyz
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        # if opt.neus_network:
        #     self.sdf_network = SDFNetwork(d_in = 3,
        #                                   d_out = 257,
        #                                   d_hidden = 256,
        #                                   n_layers = 8,
        #                                   bias = 1.0,
        #                                   multires = 6
        #                                 )               
        #     self.color_network = RenderingNetwork(d_in = 9,
        #                                         d_features = 256,
        #                                         d_out = 3,
        #                                         d_hidden = 256,
        #                                         n_layers = 4,
        #                                         multires_view = 4)

        # else:
        #     self.encoder = None
        #     if self.opt.sdf_use_pe:
        #         self.sdf_network = SDFNet(
        #                             self.opt,
        #                             encoder = self.encoder,
        #                             encoding=encoding,
        #                             num_layers=num_layers,
        #                             hidden_dim=hidden_dim,
        #                             geo_feat_dim=geo_feat_dim,
        #                             bound=bound,
        #                             kwargs=kwargs
        #                             )
        #     else:
        self.sdf_network = SDFNet(
                            self.opt,
                            encoding=encoding,
                            # encoder = self.encoder,
                            num_layers=num_layers,
                            hidden_dim=hidden_dim,
                            geo_feat_dim=geo_feat_dim,
                            bound=bound,
                            kwargs=kwargs
                            )

        self.color_network = ColorNet(self.opt,
                            encoding_dir=encoding_dir,
                            num_layers_color=num_layers_color,
                            geo_feat_dim=geo_feat_dim,
                            hidden_dim_color=hidden_dim_color,
                            bound=bound,
                            kwargs=kwargs
                            )
    
        self.deviation_network = SingleVarianceNet(init_val = 0.3)
        # self.deviation_network = SingleVarianceNet(init_val = 0.03)

        # self.anneal_end = 1000
        self.anneal_end = 0
        self.global_step = 0

    def get_deform(self,x,t):
        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        return deform

    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        deform = self.get_deform(x, t)
        x = x + deform
        print("not checked")
        exit(1)
        # sdf, gradients, geo_feat = self.sdf_network.forward_with_gradients(x)
        color = self.color_network(x, gradients, d, geo_feat)
        return sdf, gradients, color

    def sdf(self, x, t):
        # x: [N, 3], in [-bound, bound]
        deform = self.get_deform(x, t)
        x_deformed = x + deform
        sdf, geo_feat = self.sdf_network(x_deformed)
        return sdf
               
    def sdf_with_gradients(self, x, t):
        with torch.enable_grad():
            x.requires_grad_(True)
            deform = self.get_deform(x, t)
            x_deformed = x + deform
            sdf, geo_feat = self.sdf_network.forward(x_deformed)
            gradients = torch.autograd.grad(
                sdf,
                x,
                torch.ones_like(sdf, device=x.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        gradients = F.normalize(gradients, p=2, dim =-1)
        return sdf, gradients, geo_feat, deform
        # return sdf, gradients.detach(), geo_feat

    def color(self, points, normals, view_dirs, feature_vectors, mask=None):
        return self.color_network(points, normals, view_dirs, feature_vectors)

    def deviation(self, x):
        return self.deviation_network(x)

    def alpha_and_color(self, xyzs, dirs, deltas):
        total_samples = xyzs.shape[0]
        device = xyzs.device

        sdf, normals, rgbs = self(xyzs, dirs)


        # normals = F.normalize(normals, dim = -1, p=2)
        inv_s = self.deviation(torch.zeros([1, 3], device=device).to(device))[:, :1].clip(1e-6, 1e6) # Single parameter
        inv_s = inv_s.expand(total_samples, 1)
        true_cos = (dirs * normals).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        cos_anneal_ratio = self.get_cos_anneal_ratio()
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas[...,0].reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas[...,0].reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(total_samples).clip(0.0, 1.0)

        gradient_error = ((torch.linalg.norm(normals, ord=2, dim=-1) - 1.0) ** 2).mean()

        return alpha, rgbs, gradient_error


    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.global_step / self.anneal_end])
        # return 1.0

    # optimizer utils
    def get_params(self, lr, lr_net):

        params = [
            {'params': self.sdf_network.parameters(), 'lr': lr},
            {'params': self.color_network.parameters(), 'lr': lr},
            {'params': self.deviation_network.parameters(), 'lr': lr},  
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]

        # params = [
        #     {'params': self.sdf_network.parameters(), 'lr': lr},
        #     {'params': self.color_network.parameters(), 'lr': lr},
        #     {'params': self.deviation_network.parameters(), 'lr': lr},
        # ]
        # params = [
        #     {'params': self.parameters(), 'lr': lr}
        # ]
        # if self.bg_radius > 0:
        #     params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
        #     params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
