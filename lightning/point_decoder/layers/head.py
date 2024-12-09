import torch
import torch.nn as nn
from typing import Callable, Dict
from .gaussian_renderer import render
from .activation import TruncExp, Normalize
from .normalization import CustomGroupNorm1d
from simple_knn._C import distCUDA2


def positional_encoding(x, n_freqs):
    f = (2**torch.arange(n_freqs, dtype=torch.float32, device=x.device))
    fx = torch.flatten(f[None, None, :, None] * x[:, :, None, :], -2, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


class GaussianHead(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        sh_degree: int = 0, 
        white_bg: bool = True, 
        scale_shift: int = 0,
        scale_activation: str = "relu",
        weight_init_gains: Dict[str, float] = {},
        bias_init_values: Dict[str, float] = {},
        n_freqs: int = 0,
        use_position_for_scale: bool = False,
        autocast: bool = False,
        pos_activation: str = "tanh",
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = CustomGroupNorm1d
    ) -> None:
        super().__init__()
        assert sh_degree <= 3, "sh_degree must be less than or equal to 3"
        assert len(weight_init_gains.keys()) == len(bias_init_values.keys()) >= 3
        assert scale_activation in ["relu", "exp"]

        self.n_freqs = n_freqs
        self.in_features = in_features if n_freqs <= 0 else in_features * 2 * self.n_freqs
        self.hidden_features = self.in_features
        self.sh_degree = sh_degree
        self.num_coeffs = 3 * (sh_degree + 1)**2
        self.out_features = self.num_coeffs + 1 + 3 + 4  # sh + opacity + scale + rotation
        self.white_bg = white_bg
        self.scale_shift = scale_shift
        self.use_position_for_scale = use_position_for_scale
        self.autocast = autocast

        self.norm = None if norm_layer is None \
                    else norm_layer(in_features)
        
        self.feat2attr = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            act_layer(),
            nn.Linear(self.hidden_features, self.out_features),
        )

        if pos_activation == "tanh":
            raise NotImplementedError
            # self.pos_activation = nn.Tanh()
        else:
            self.pos_activation = nn.Identity()

        self.act_alpha = nn.Sigmoid()
        if use_position_for_scale:
            self.act_scale = nn.Identity()
        else:
            if scale_activation == "relu":
                self.act_scale = nn.ReLU()
            else:
                self.act_scale = TruncExp(self.scale_shift)
        self.act_rotation = Normalize()

        # self._initialize_networks(weight_init_gains, bias_init_values)
        self.register_buffer("background", torch.tensor([1., 1., 1.] if white_bg else [0., 0., 0.]))
    
    def _initialize_networks(self, weight_init_gains, bias_init_values):
        def initialize_layer(weight, weight_init_gain, bias=None, bias_init_value=0.):
            nn.init.xavier_uniform_(weight, weight_init_gain)
            if bias is not None:
                nn.init.constant_(bias, bias_init_value)
        initialize_layer(self.feat2attr[-1].weight[:self.num_coeffs, :], weight_init_gains["sh"], self.feat2attr[-1].bias[:self.num_coeffs], bias_init_values["sh"])
        initialize_layer(self.feat2attr[-1].weight[self.num_coeffs:self.num_coeffs+1, :], weight_init_gains["opacity"], self.feat2attr[-1].bias[self.num_coeffs:self.num_coeffs+1], bias_init_values["opacity"])
        initialize_layer(self.feat2attr[-1].weight[self.num_coeffs+1:self.num_coeffs+1+3, :], weight_init_gains["scale"], self.feat2attr[-1].bias[self.num_coeffs+1:self.num_coeffs+1+3], bias_init_values["scale"])
        initialize_layer(self.feat2attr[-1].weight[self.num_coeffs+1+3:self.num_coeffs+1+3+4, :], weight_init_gains["rotation"], self.feat2attr[-1].bias[self.num_coeffs+1+3:self.num_coeffs+1+3+4], bias_init_values["rotation"])
    
    def forward(self, positions, features, viewpoint_cameras):
        # batch, number of points
        B, N = features.shape[:-1]

        # normalize
        if self.norm is not None:
            features = self.norm(features)

        if self.n_freqs > 0:
            features = positional_encoding(features, n_freqs=self.n_freqs)
        
        # decode features into Gaussian attributes
        attributes = self.feat2attr(features)

        # unbind
        colors = [None] * B
        coeffs = attributes[..., :self.num_coeffs].view(B, N, (self.sh_degree+1)**2, 3)
        alphas = self.act_alpha(attributes[..., self.num_coeffs:self.num_coeffs+1])
        scales = self.act_scale(attributes[..., self.num_coeffs+1:self.num_coeffs+1+3])
        rotations = self.act_rotation(attributes[..., self.num_coeffs+1+3:self.num_coeffs+1+3+4])
    
        # render images
        images = []
        n_views = len(viewpoint_cameras.FoVx[0])  # (B, V)
        for i, (position, coeff, color, alpha, scale, rotation) in enumerate(
            zip(positions, coeffs, colors, alphas, scales, rotations)):
            
            if self.use_position_for_scale:
                dist2 = torch.clamp_min(distCUDA2(position.detach()), 0.0000001)
                scale_anchor = torch.sqrt(dist2)[...,None].repeat(1, 3)
                scale = torch.exp(torch.log(scale_anchor) * scale - self.scale_shift)

            images_i = []
            for j in range(n_views):
                render_pkg = render(
                    viewpoint_cameras.FoVx[i, j],
                    viewpoint_cameras.FoVy[i, j], 
                    viewpoint_cameras.image_width[i, j], 
                    viewpoint_cameras.image_height[i, j], 
                    viewpoint_cameras.world_view_transform[i, j], 
                    viewpoint_cameras.full_proj_transform[i, j], 
                    viewpoint_cameras.camera_center[i, j], 
                    position, 
                    coeff, 
                    alpha, 
                    scale, 
                    rotation, 
                    self.background, 
                    self.sh_degree, 
                    override_color=color,
                    autocast=self.autocast,
                    debug=False
                )
                images_i.append(render_pkg["render"])
            images.append(torch.stack(images_i))
        images = torch.stack(images)

        return positions, images


# class GaussianHead(nn.Module):
#     def __init__(
#         self, 
#         in_features: int, 
#         sh_degree: int = 0, 
#         white_bg: bool = True, 
#         scale_shift: int = 0,
#         scale_activation: str = "relu",
#         weight_init_gains: Dict[str, float] = {},
#         bias_init_values: Dict[str, float] = {},
#         n_freqs: int = 0,
#         use_position_for_scale: bool = False,
#         autocast: bool = False,
#         pos_activation: str = "tanh",
#         act_layer: Callable[..., nn.Module] = nn.GELU,
#         norm_layer: Callable[..., nn.Module] = CustomGroupNorm1d
#     ) -> None:
#         super().__init__()
#         assert sh_degree <= 3, "sh_degree must be less than or equal to 3"
#         assert len(weight_init_gains.keys()) == len(bias_init_values.keys()) >= 3
#         assert scale_activation in ["relu", "exp"]

#         self.n_freqs = n_freqs
#         self.in_features = in_features if n_freqs <= 0 else in_features * 2 * self.n_freqs
#         self.hidden_features = self.in_features
#         self.sh_degree = sh_degree
#         self.num_coeffs = 3 * (sh_degree + 1)**2
#         self.out_features = 3 + self.num_coeffs + 1 + 3 + 4  # xyz + sh + opacity + scale + rotation
#         self.white_bg = white_bg
#         self.scale_shift = scale_shift
#         self.use_position_for_scale = use_position_for_scale
#         self.autocast = autocast

#         self.norm = None if norm_layer is None \
#                     else norm_layer(in_features)
        
#         self.feat2attr = nn.Sequential(
#             nn.Linear(self.in_features, self.hidden_features),
#             act_layer(),
#             nn.Linear(self.hidden_features, self.out_features),
#         )

#         if pos_activation == "tanh":
#             raise NotImplementedError
#             # self.pos_activation = nn.Tanh()
#         else:
#             self.pos_activation = nn.Identity()

#         self.act_alpha = nn.Sigmoid()
#         if use_position_for_scale:
#             self.act_scale = nn.Identity()
#         else:
#             if scale_activation == "relu":
#                 self.act_scale = nn.ReLU()
#             else:
#                 self.act_scale = TruncExp(self.scale_shift)
#         self.act_rotation = Normalize()

#         self.register_buffer("background", torch.tensor([1., 1., 1.] if white_bg else [0., 0., 0.]))
    
#     def _initialize_networks(self, weight_init_gains, bias_init_values):
#         raise NotImplementedError
#         def initialize_layer(weight, weight_init_gain, bias=None, bias_init_value=0.):
#             nn.init.xavier_uniform_(weight, weight_init_gain)
#             if bias is not None:
#                 nn.init.constant_(bias, bias_init_value)
#         initialize_layer(self.feat2attr[-1].weight[:self.num_coeffs, :], weight_init_gains["sh"], self.feat2attr[-1].bias[:self.num_coeffs], bias_init_values["sh"])
#         initialize_layer(self.feat2attr[-1].weight[self.num_coeffs:self.num_coeffs+1, :], weight_init_gains["opacity"], self.feat2attr[-1].bias[self.num_coeffs:self.num_coeffs+1], bias_init_values["opacity"])
#         initialize_layer(self.feat2attr[-1].weight[self.num_coeffs+1:self.num_coeffs+1+3, :], weight_init_gains["scale"], self.feat2attr[-1].bias[self.num_coeffs+1:self.num_coeffs+1+3], bias_init_values["scale"])
#         initialize_layer(self.feat2attr[-1].weight[self.num_coeffs+1+3:self.num_coeffs+1+3+4, :], weight_init_gains["rotation"], self.feat2attr[-1].bias[self.num_coeffs+1+3:self.num_coeffs+1+3+4], bias_init_values["rotation"])
    
#     def forward(self, positions, features, viewpoint_cameras):
#         # batch, number of points
#         B, N = features.shape[:-1]

#         # normalize
#         if self.norm is not None:
#             features = self.norm(features)

#         if self.n_freqs > 0:
#             features = positional_encoding(features, n_freqs=self.n_freqs)
        
#         # decode features into Gaussian attributes
#         attributes = self.feat2attr(features)

#         # unbind
#         colors = [None] * B
#         positions = torch.tanh(attributes[..., :3]) * 0.45
#         coeffs = attributes[..., 3:3+self.num_coeffs].view(B, N, (self.sh_degree+1)**2, 3)
#         alphas = self.act_alpha(attributes[..., 3+self.num_coeffs:3+self.num_coeffs+1])
#         scales = self.act_scale(attributes[..., 3+self.num_coeffs+1:3+self.num_coeffs+1+3])
#         rotations = self.act_rotation(attributes[..., 3+self.num_coeffs+1+3:3+self.num_coeffs+1+3+4])
    
#         # render images
#         images = []
#         n_views = len(viewpoint_cameras.FoVx[0])  # (B, V)
#         for i, (position, coeff, color, alpha, scale, rotation) in enumerate(
#             zip(positions, coeffs, colors, alphas, scales, rotations)):
            
#             if self.use_position_for_scale:
#                 dist2 = torch.clamp_min(distCUDA2(position.detach()), 0.0000001)
#                 scale_anchor = torch.sqrt(dist2)[...,None].repeat(1, 3)
#                 scale = torch.exp(torch.log(scale_anchor) * scale - self.scale_shift)

#             images_i = []
#             for j in range(n_views):
#                 render_pkg = render(
#                     viewpoint_cameras.FoVx[i, j],
#                     viewpoint_cameras.FoVy[i, j], 
#                     viewpoint_cameras.image_width[i, j], 
#                     viewpoint_cameras.image_height[i, j], 
#                     viewpoint_cameras.world_view_transform[i, j], 
#                     viewpoint_cameras.full_proj_transform[i, j], 
#                     viewpoint_cameras.camera_center[i, j], 
#                     position, 
#                     coeff, 
#                     alpha, 
#                     scale, 
#                     rotation, 
#                     self.background, 
#                     self.sh_degree, 
#                     override_color=color,
#                     autocast=self.autocast,
#                     debug=False
#                 )
#                 images_i.append(render_pkg["render"])
#             images.append(torch.stack(images_i))
#         images = torch.stack(images)

#         return positions, images
