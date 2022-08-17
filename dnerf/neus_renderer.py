import math
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import raymarching
from .utils import custom_meshgrid

from plyfile import PlyData, PlyElement
import numpy  as np
def write_ply(points, filename, text=False):
    """
    input: Nx3, write points to filename as PLY format.
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)
        

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples



def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    sphere = trimesh.creation.icosphere(radius=1)
    trimesh.Scene([pc, axes, sphere]).show()


class NeuSRenderer(nn.Module):
    def __init__(self,
                 opt,
                 bound=1,
                 cuda_ray=False,
                 density_scale=1, # scale up deltas (or sigmas), to make the sdf grid more sharp. larger value than 1 usually improves performance.
                 min_near=0.2,
                 density_thresh=0.01,
                 bg_radius=-1,
                 ):
        super().__init__()

        self.up_sample_steps = 4 # Neus upsample step

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate sdf grid and hashing.
        # if self.opt.smpl_guided_sample:
        #     import trimesh
        #     mesh = trimesh.load(opt.smpl_guided_sample)

        #     world_vex = torch.tensor(mesh.vertices)
        #     min_xyz = world_vex.min(dim=0)[0]
        #     max_xyz = world_vex.max(dim=0)[0]

        #     # min_xyz -= 0.250
        #     min_xyz -= 0.100
        #     # max_xyz += 0.250
        #     max_xyz += 0.100

        #     world_bbox = torch.cat([min_xyz,max_xyz], dim = 0).float()
        #     aabb_train = world_bbox
        #     aabb_infer = world_bbox.clone()
        # else:
        #     aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        #     aabb_infer = aabb_train.clone()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.time_size = 64
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius # radius of the background sphere.

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid (with an extra time dimension)
            density_grid = torch.zeros(self.time_size, self.cascade, self.grid_size ** 3) # [T, CAS, H * H * H]
            density_bitfield = torch.zeros(self.time_size, self.cascade * self.grid_size ** 3 // 8, dtype=torch.uint8) # [T, CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # time stamps for density grid
            times = ((torch.arange(self.time_size, dtype=torch.float32) + 0.5) / self.time_size).view(-1, 1, 1) # [T, 1, 1]
            self.register_buffer('times', times)
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0


    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def sdf(self, x):
        raise NotImplementedError()

    def sdf_with_gradients(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def deviation(self, x):
        raise NotImplementedError()

    def alpha_and_color(self, x ,d):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return 
        # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        device = rays_o.device
        # pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        # pts = torch.min(torch.max(pts, self.aabb[:3]), self.aabb[3:]) # a manual clip.
        # radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        # inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        # inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        # cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        cos_val = cos_val.clip(-1e3, 0.0)

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1],device=device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, time, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        pts = torch.min(torch.max(pts, self.aabb[:3]), self.aabb[3:]) # a manual clip.
        
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf(pts.reshape(-1, 3), time).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def run(self, rays_o, rays_d, time, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1] # [B,N]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        batch_size = N
        n_samples = num_steps + upsample_steps

        # choose aabb
        self.aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        #print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0) # [1, T]
        z_vals = z_vals.expand((N, num_steps)) # [N, T]
        z_vals = nears + (fars - nears) * z_vals # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            #z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # upsample z_vals (Neus-like)
        if upsample_steps > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = torch.min(torch.max(pts, self.aabb[:3]), self.aabb[3:]) # a manual clip.

                sdf = self.sdf(pts.reshape(-1, 3), time).reshape(batch_size, num_steps)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                upsample_steps // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  time,
                                                  last=(i + 1 == self.up_sample_steps))



        deltas = z_vals[..., 1:] - z_vals[..., :-1] # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

        mid_z_vals = z_vals + deltas * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        pts = torch.min(torch.max(pts, self.aabb[:3]), self.aabb[3:]) # a manual clip.
        # write_ply(pts.reshape(-1,3),'test_sample.ply')
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        sdf, gradients, feature_vector, deform = self.sdf_with_gradients(pts, time)
        # sdf, feature_vector = self.sdf_network(pts)
        # gradients = torch.ones_like(pts,device=device)


        inv_s = self.deviation(torch.zeros([1, 3], device=device).to(device))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        can_pts = pts + deform

        if self.opt.canonical_direction:
                # dirs = self.feature_net.vector_posed_to_canonical(dirs,can_pts)
            can_pts = can_pts.reshape(batch_size, n_samples, 3)
            dirs = can_pts[:,1:,:] - can_pts[:,0:-1,:]
            dirs += 1e-7
            dirs = torch.cat([dirs,dirs[:,-1,:].unsqueeze(1)], dim = 1)
            dirs = F.normalize(dirs,p=2,dim=-1).reshape(-1,3)
            can_pts = can_pts.reshape(-1, 3)
            # can_dists = torch.norm(dirs, p=2, dim=-1)
            # dirs = dirs / can_dists.unsqueeze(-1)

        # true_cos = (dirs * gradients).sum(-1, keepdim=True)
        true_cos = (dirs * F.normalize(gradients,p=2,dim=-1)).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        cos_anneal_ratio = self.get_cos_anneal_ratio()
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * deltas.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * deltas.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1], device = device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        # gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        gradient_error = gradient_error.mean()
        # mask = weights > 1e-4 # hard coded

        # sampled_color = self.color(pts, gradients, dirs, feature_vector, mask = mask.reshape(-1)).reshape(batch_size, n_samples, 3)
        # sampled_color = self.color(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)
        # sampled_color = self.color(pts, F.normalize(gradients,p=2,dim=-1), dirs, feature_vector).reshape(batch_size, n_samples, 3)
        # sampled_color = self.color(can_pts, F.normalize(gradients,p=2,dim=-1), torch.ones_like(dirs), feature_vector).reshape(batch_size, n_samples, 3)
        sampled_color = self.color(can_pts, F.normalize(gradients,p=2,dim=-1), dirs, feature_vector).reshape(batch_size, n_samples, 3)
        # sampled_color = self.color(can_pts, F.normalize(gradients,p=2,dim=-1), feature_vector).reshape(batch_size, n_samples, 3)

        # from torchviz import make_dot

        # dot = make_dot(sampled_color.mean(), params=dict(self.named_parameters()), show_attrs=True, show_saved=True)
        # dot.format = 'png'
        # dot.render('torchviz-sample')
        # exit(1)
        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = (sampled_color * weights[:, :, None]).sum(dim=1)
        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d.reshape(-1, 3)) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            # bg_color = 0
            

        image = image + (1 - weights_sum).unsqueeze(0) * bg_color
        # image = image
        # torch.where(weights_sum!=0)
        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360
        # z_vals_shifted = torch.cat([z_vals[..., 1:], sample_dist * torch.ones_like(z_vals[..., :1])], dim=-1)
        # mid_zs = (z_vals + z_vals_shifted) / 2 # [N, T]
        # loss_dist = (torch.abs(mid_zs.unsqueeze(1) - mid_zs.unsqueeze(2)) * (weights.unsqueeze(1) * weights.unsqueeze(2))).sum() + 1/3 * ((z_vals_shifted - z_vals_shifted) * (weights ** 2)).sum()
        return {
            'depth': depth,
            'image': image,
            'eikonal_loss': gradient_error,
            'deform': deform,
            # 'weights_sum': weights_sum.mean(),
            'weights_sum': weights_sum,
            'weights_max': torch.max(weights, dim=-1, keepdim=True)[0].mean(),
        }

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        N = rays_o.shape[0] # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near)

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d) # [N, 3]
        elif bg_color is None:
            bg_color = 1
            # bg_color = 0

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_() # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, counter, self.mean_count, perturb, 128, force_all_rays, dt_gamma, max_steps)

            # # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
            # # sigmas = density_outputs['sigma']
            # # rgbs = self.color(xyzs, dirs, **density_outputs)
            # sigmas = self.density_scale * sigmas

            alpha, rgbs, gradient_error = self.alpha_and_color(xyzs, dirs, deltas)

            # #print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            # special case for CCNeRF's residual learning
            if len(alpha.shape) == 2:
                K = alpha.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train_neus(alpha[k], rgbs[k], deltas, rays)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))
            
                depth = torch.stack(depths, axis=0) # [K, B, N]
                image = torch.stack(images, axis=0) # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train_neus(alpha, rgbs, deltas, rays)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                # image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                depth = torch.clamp(depth - nears, min=0) / (fars - nears)
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)

        else:
           
            # allocate outputs 
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            #dtype = torch.half if torch.is_autocast_enabled() else torch.float32
            # output should always be float32! only network inference uses half.
            dtype = torch.float32
            
            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            
            n_alive = N
            alive_counter = torch.zeros([1], dtype=torch.int32, device=device)

            rays_alive = torch.zeros(2, n_alive, dtype=torch.int32, device=device) # 2 is used to loop old/new
            rays_t = torch.zeros(2, n_alive, dtype=dtype, device=device)

            step = 0
            i = 0
            while step < 1024: # hard coded max step

                # count alive rays 
                if step == 0:
                    # init rays at first step.
                    torch.arange(n_alive, out=rays_alive[0])
                    rays_t[0] = nears
                else:
                    alive_counter.zero_()
                    raymarching.compact_rays(n_alive, rays_alive[i % 2], rays_alive[(i + 1) % 2], rays_t[i % 2], rays_t[(i + 1) % 2], alive_counter)
                    n_alive = alive_counter.item() # must invoke D2H copy here
                
                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)

                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb, dt_gamma, max_steps)

                # sigmas, rgbs = self(xyzs, dirs)
                # # density_outputs = self.density(xyzs) # [M,], use a dict since it may include extra things, like geo_feat for rgb.
                # # sigmas = density_outputs['sigma']
                # # rgbs = self.color(xyzs, dirs, **density_outputs)
                # sigmas = self.density_scale * sigmas

                alpha, rgbs, gradient_error = self.alpha_and_color(xyzs, dirs, deltas)

                raymarching.composite_rays_neus(n_alive, n_step, rays_alive[i % 2], rays_t[i % 2], alpha, rgbs, deltas, weights_sum, depth, image)

                #print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step
                i += 1

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            # image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        return {
            'depth': depth,
            'image': image,
            'eikonal_loss': gradient_error,
            'weights_sum': weights_sum,
        }

    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[0]
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)

        count = torch.zeros_like(self.sdf_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.sdf_grid[count == 0] = -1

        #print(f'[mark untrained grid] {(count == 0).sum()} from {resolution ** 3 * self.cascade}')

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        ### update sdf grid

        tmp_grid = - torch.ones_like(self.sdf_grid)
        
        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.sdf_grid.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query sdf
                            sdf = self.sdf(cas_xyzs).reshape(-1).detach()
                            # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                            # scale == 2 * sqrt(3) / 1024
                            # sdf *= self.density_scale * 0.003383
                            # sdf = torch.abs(sdf * 0.003383).float()
                            # sdf = torch.abs(sdf).float()
                            # inv_s = self.deviation(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)
                            # sigmoid_sdf = torch.sigmoid(sdf*inv_s)
                            # sdf = sigmoid_sdf * (1 - sigmoid_sdf)
                            inv_s = self.deviation(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)
                            sigmoid_sdf = torch.sigmoid(sdf*inv_s)
                            sdf = inv_s * sigmoid_sdf * (1 - sigmoid_sdf)
                            sdf = torch.exp(sdf)
                            # sdf = 0.01 - sdf*sdf
                            # sdf = 1 - sdf * sdf * inv_s
                            # assign 
                            tmp_grid[cas, indices] = sdf.float()

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 2
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.sdf_grid.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.sdf_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.sdf_grid.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sdf = self.sdf(cas_xyzs).reshape(-1).detach()
                # from `scalbnf(MIN_CONE_STEPSIZE(), 0)`, check `splat_grid_samples_nerf_max_nearest_neighbor`
                # scale == 2 * sqrt(3) / 1024
                # sigmas *= self.density_scale * 0.003383
                # # assign 
                # tmp_grid[cas, indices] = sigmas
                # sdf = torch.abs(sdf * 0.003383).float()
                # sdf = torch.abs(sdf).float()
                inv_s = self.deviation(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)
                sigmoid_sdf = torch.sigmoid(sdf*inv_s)
                # sdf = sigmoid_sdf * (1 - sigmoid_sdf) * 0.003383
                # sdf = sigmoid_sdf * (1 - sigmoid_sdf)
                sdf = inv_s * sigmoid_sdf * (1 - sigmoid_sdf)
                sdf = torch.exp(sdf)
                # import pdb
                # pdb.set_trace()
                # assign 
                # sdf = 0.01 - sdf*sdf * inv_s
                # sdf = 1 - sdf * sdf * inv_s
                tmp_grid[cas, indices] = sdf.float()
        # ema update
        valid_mask = (self.sdf_grid >= 0) & (tmp_grid >= 0)
        self.sdf_grid[valid_mask] = torch.maximum(self.sdf_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.sdf_grid.clamp(min=0)).item()
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        # density_thresh = self.density_thresh
        self.density_bitfield = raymarching.packbits(self.sdf_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        #print(f'[sdf grid] min={self.sdf_grid.min().item():.4f}, max={self.sdf_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.sdf_grid > 0.01).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')


    def render(self, rays_o, rays_d, time, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b+1, head:tail], rays_d[b:b+1, head:tail], time[b:b+1], **kwargs)
                    depth[b:b+1, head:tail] = results_['depth']
                    image[b:b+1, head:tail] = results_['image']
                    head += max_ray_batch
            
            results = {}
            results['depth'] = depth
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, time, **kwargs)

        return results