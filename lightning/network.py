import torch,timm,random
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd.functional import vjp
import numpy as np

from PIL import Image

# from lightning.renderer_2dgs import Renderer
from lightning.renderer import Renderer
from lightning.utils import MiniCam
from tools.rsh import rsh_cart_3

import pytorch_lightning as L
from torchvision import transforms

from functools import partial
from .point_decoder.utils.structure import Point
from .point_decoder.utils.modules import PointModule, PointSequential
from .point_decoder.autoencoder import *
from .point_decoder.utils.misc import offset2batch, batch2offset

# from simple_knn._C import distCUDA2

# @torch.no_grad()
# def _activation_scale(x):
#     dist2 = torch.clamp_min(distCUDA2(x), 0.0000001)
#     scale = torch.sqrt(dist2)[..., None].repeat(1, 2)
#     return scale

def group_cat(
    tensors,
    index,
    dim: int = 0,
    return_index: bool = False,
):
    assert len(tensors) == len(index)
    index, perm = torch.cat(index).sort(stable=True)
    out = torch.cat(tensors, dim)[perm]
    return (out, index) if return_index else out

class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))

        return outputs[:,1:]
    
    def freeze(self, is_train:bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err

class GroupAttBlock(L.LightningModule):
    def __init__(self, inner_dim: int, cond_dim: int, 
                 num_heads: int, eps: float,
                attn_drop: float = 0., attn_bias: bool = False,
                mlp_ratio: float = 2., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.cnn = nn.Conv3d(inner_dim, inner_dim, kernel_size=3, padding=1, bias=False)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        
    def forward(self, x, cond, group_axis, block_size):
        # x: [B, C, D, H, W]
        # cond: [B, L_cond, D_cond]

        B,C,D,H,W = x.shape

        # Unfold the tensor into patches
        patches = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size).unfold(4, block_size, block_size)
        patches = patches.reshape(B, C, -1, block_size**3)
        patches = torch.einsum('bcgl->bglc',patches).reshape(B*group_axis**3, block_size**3,C)
     
        # cross attention
        patches = patches + self.cross_attn(self.norm1(patches), cond, cond, need_weights=False)[0]
        patches = patches + self.mlp(self.norm2(patches))

        # 3D CNN
        patches = self.norm3(patches)
        patches = patches.view(B, group_axis,group_axis,group_axis,block_size,block_size,block_size,C) 
        patches = torch.einsum('bdhwzyxc->bcdzhywx',patches).reshape(x.shape)
        patches = patches + self.cnn(patches)

        return patches
    

class VolTransformer(L.LightningModule):
    """
    Transformer with condition and modulation that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(self, embed_dim: int, image_feat_dim: int, n_groups: list,
                 vol_low_res: int, vol_high_res: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6):
        super().__init__()

        # attributes
        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.block_size = [vol_low_res//item for item in n_groups]
        self.embed_dim = embed_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, vol_low_res,vol_low_res,vol_low_res) * (1. / embed_dim) ** 0.5)
        self.layers = nn.ModuleList([
            GroupAttBlock(
                inner_dim=embed_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.deconv = nn.ConvTranspose3d(embed_dim, out_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, image_feats):
        # image_feats: [B, N_views, C, DHW]
        # camera_embeddings: [N, D_mod]
        
        B,V,C,D,H,W = image_feats.shape
        
        volume_feats = []
        for n_group in self.n_groups:
            block_size = D//n_group
            blocks = image_feats.unfold(3, block_size, block_size).unfold(4, block_size, block_size).unfold(5, block_size,block_size)
            blocks = blocks.contiguous().view(B,V,C,n_group**3,block_size**3)
            blocks = torch.einsum('bvcgl->bgvlc',blocks).reshape(B*n_group**3,block_size**3*V,C)
            volume_feats.append(blocks)

        x = self.pos_embed.repeat(B, 1,1,1,1)  # [N, G, L, D]

        for i, layer in enumerate(self.layers):
            group_idx = i%len(self.block_size)
            x = layer(x, volume_feats[group_idx], self.n_groups[group_idx], self.block_size[group_idx])

        x = self.norm(torch.einsum('bcdhw->bdhwc',x))
        x = torch.einsum('bdhwc->bcdhw',x)

        # separate each plane and apply deconv
        x_up = self.deconv(x)  # [3*N, D', H', W']
        x_up = torch.einsum('bcdhw->bdhwc',x_up).contiguous()
        return x_up


def get_pose_feat(src_exts, tar_ext, src_ixts, W, H):
    """
    src_exts: [B,N,4,4]
    tar_ext: [B,4,4]
    src_ixts: [B,N,3,3]
    """

    B = src_exts.shape[0]
    c2w_ref = src_exts[:,0].view(B,-1)
    normalize_facto = torch.tensor([W,H]).unsqueeze(0).to(c2w_ref)
    fx_fy = src_ixts[:,0,[0,1],[0,1]]/normalize_facto
    cx_cy = src_ixts[:,0,[0,1],[2,2]]/normalize_facto

    return torch.cat((c2w_ref,fx_fy,fx_fy), dim=-1)

def projection(grid, w2cs, ixts):

    points = grid.reshape(1,-1,3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]


class ModLN(L.LightningModule):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
    
class Decoder(L.LightningModule):
    def __init__(self, in_dim, sh_dim, scaling_dim, rotation_dim, opacity_dim, K=1, latent_dim=256):
        super(Decoder, self).__init__()

        self.K = K
        self.sh_dim = sh_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim  = rotation_dim
        self.out_dim = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim

        num_layer = 2
        layers_coarse = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                 [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer-1) + \
                 [nn.Linear(in_dim, self.out_dim*K)]
        # add one more layer for feas -> attributes
        self.mlp_coarse = nn.Sequential(*layers_coarse)


        cond_dim = 8
        self.norm = nn.LayerNorm(in_dim)
        self.feature_dim = in_dim
        self.cross_att = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=16, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        layers_fine = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                  [nn.Linear(in_dim, in_dim + self.sh_dim)]
        self.mlp_fine = nn.Sequential(*layers_fine)
        # layers_shs = [nn.Linear(256, self.sh_dim)]
        # self.shs_fine =  nn.Sequential(*layers_shs)
        
        self.init(self.mlp_coarse)
        self.init(self.mlp_fine)
        # self.init(self.shs_fine)

    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue 
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    
    def forward_coarse(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        parameters = parameters.view(*parameters.shape[:-1],self.K,-1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters, 
            [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
            )
        opacity = opacity + opacity_shift 
        scaling = scaling + scaling_shift 
        offset = torch.sigmoid(offset)*2-1.0

        B = opacity.shape[0]
        sh = sh.view(B,-1,self.sh_dim//3,3)
        opacity = opacity.view(B,-1,self.opacity_dim)
        scaling = scaling.view(B,-1,self.scaling_dim)
        rotation = rotation.view(B,-1,self.rotation_dim)
        offset = offset.view(B,-1,3)
        return offset, sh, scaling, rotation, opacity

    def forward_fine(self, volume_feat, point_feats):
        volume_feat = self.norm(volume_feat.unsqueeze(1))
        x = self.cross_att(volume_feat, point_feats, point_feats, need_weights=False)[0]
        features = self.mlp_fine(x).float()
        return features[..., :self.feature_dim], features[..., self.feature_dim:]
    
class Network(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # modules
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=True,
        )
        
        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16*2, eps=1e-6)

        # build volume position
        self.grid_reso = cfg.model.vol_embedding_reso
        # self.register_buffer("dense_grid", self.build_dense_grid(self.grid_reso))
        # self.register_buffer("centers", self.build_dense_grid(self.grid_reso*2))

        # view embedding
        if cfg.model.view_embed_dim > 0:
            self.view_embed = nn.Parameter(torch.randn(1, 4, cfg.model.view_embed_dim,1,1,1) * (1. / cfg.model.view_embed_dim) ** 0.5)
        
        # build volume transformer
        self.n_groups = cfg.model.n_groups
        vol_embedding_dim = cfg.model.embedding_dim
        self.vol_decoder = VolTransformer(
            embed_dim=vol_embedding_dim, image_feat_dim=encoder_feat_dim+cfg.model.view_embed_dim,
            vol_low_res=self.grid_reso, vol_high_res=self.grid_reso*2, out_dim=cfg.model.vol_embedding_out_dim, n_groups=self.n_groups,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        )
        self.feat_vol_reso = cfg.model.vol_feat_reso
        self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))
        
        # grouping configuration
        self.n_offset_groups = cfg.model.n_offset_groups
        self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*2))
        self.group_centers = self.group_centers.reshape(1,-1,3)

        # 2DGS model OR 3DGS model
        self.sh_dim = (cfg.model.sh_degree+1)**2*3
        self.scaling_dim, self.rotation_dim = 3, 4
        self.opacity_dim = 1
        self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        self.K = cfg.model.K
        vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        self.decoder = Decoder(vol_embedding_out_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K)
        self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        self.opacity_shift = -2.1792
        self.voxel_size = 2.0/(self.grid_reso*2)
        self.scaling_shift = np.log(0.5*self.voxel_size/3.0)
        self.fine_scaling_shift = np.log(0.5*self.voxel_size/(8*3.0))

        # gradient masking nums
        self.k_num = cfg.model.k_num

        # point decoder parameters
        self.pdnorm_bn = cfg.model.pdnorm_bn
        self.pdnorm_ln = cfg.model.pdnorm_ln
        # global pooling
        self.enable_ada_lnnorm = cfg.model.enable_ada_lnnorm
        # upscale block
        self.enable_upscale_drop_path = cfg.model.enable_upscale_drop_path
        # mask block
        # gaussian head
        self.sh_degree = cfg.model.sh_degree
        self.enable_residual_attribute = cfg.model.enable_residual_attribute
        self.shuffle_orders = cfg.model.shuffle_orders
        self.order = [cfg.model.order] if isinstance(cfg.model.order, str) else cfg.model.order
        # slices to access each attribute
        self.num_sh = 3 * (cfg.model.sh_degree + 1)**2
        self.slice_sh = slice(0, self.num_sh)
        self.slice_opacity = slice(self.num_sh, self.num_sh+1)
        self.slice_scale = slice(self.num_sh+1, self.num_sh+1+self.scaling_dim)
        self.slice_rotation = slice(self.num_sh+1+self.scaling_dim, self.num_sh+1+self.scaling_dim+self.rotation_dim)
        # activation layers
        act_layer = nn.GELU

        # norm layers
        if self.pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=cfg.model.pdnorm_affine
                ),
                conditions=cfg.model.pdnorm_conditions,
                decouple=cfg.model.pdnorm_decouple,
                adaptive=cfg.model.pdnorm_adaptive,
            )
        else:
            bn_layer = partial(
                nn.BatchNorm1d, 
                eps=1e-3, 
                momentum=0.01, 
                track_running_stats=False,
                affine=cfg.model.bnnorm_affine
            )
        if self.pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=cfg.model.pdnorm_affine),
                conditions=cfg.model.pdnorm_conditions,
                decouple=cfg.model.pdnorm_decouple,
                adaptive=cfg.model.pdnorm_adaptive,
            )
        else:
            ln_layer = partial(
                nn.LayerNorm, 
                elementwise_affine=cfg.model.lnnorm_affine
            )

        # decoder
        dec_drop_path = [
            x.item() for x in torch.linspace(0, cfg.model.drop_path, sum(cfg.model.dec_depths))
        ][::-1]
        self.dec = nn.ModuleList([])
        # self.head = nn.ModuleList([])
        for s in range(len(cfg.model.dec_channels)):
            dec = PointSequential()

            if s > 0:
                if self.enable_residual_attribute:
                    dec.add(
                        SerializationResModule(
                            stride=cfg.model.stride[s-1],
                            order=self.order,
                            shuffle_orders=self.shuffle_orders,
                            enable_residual_attribute=self.enable_residual_attribute
                        ),
                        name="serialization"
                    )
                else:
                    dec.add(
                        SerializationModule(
                            stride=cfg.model.stride[s-1],
                            order=self.order,
                            shuffle_orders=self.shuffle_orders,
                            enable_residual_attribute=self.enable_residual_attribute
                        ),
                        name="serialization"
                    )
            else:
                if self.enable_ada_lnnorm:
                    dec.add(
                        GlobalPooling(),
                        name="global"
                    )
            
            # if not cfg.model.use_mask:
            #     cfg.model.non_leaf_ratio = [1.0 for _ in range(len(cfg.model.non_leaf_ratio))]

            # attention block
            dec_drop_path_ = dec_drop_path[
                sum(cfg.model.dec_depths[:s]) : sum(cfg.model.dec_depths[:s+1])
            ]
            for i in range(cfg.model.dec_depths[s]):
                dec.add(
                    Block(
                        channels=cfg.model.dec_channels[s],
                        num_heads=cfg.model.dec_num_head[s],
                        patch_size=cfg.model.dec_patch_size[s],
                        mlp_ratio=cfg.model.mlp_ratio,
                        qkv_bias=cfg.model.qkv_bias,
                        qk_scale=cfg.model.qk_scale,
                        attn_drop=cfg.model.attn_drop,
                        proj_drop=cfg.model.proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=cfg.model.pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=cfg.model.enable_rpe,
                        enable_flash=cfg.model.enable_flash,
                        upcast_attention=cfg.model.upcast_attention,
                        upcast_softmax=cfg.model.upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            
            # upscale module
            if self.enable_residual_attribute:
                dec.add(
                    UpscaleResModule(
                        is_first=(s==0),
                        in_channels=cfg.model.dec_channels[s],
                        out_channels=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        upscale_factor=cfg.model.upscale_factor[s],
                        n_frequencies=cfg.model.n_frequencies,
                        drop_path=dec_drop_path_[-1] \
                        if self.enable_upscale_drop_path else 0.0,
                        enable_absolute_pe=cfg.model.enable_absolute_pe,
                        enable_residual_attribute=self.enable_residual_attribute,
                        norm_layer=ln_layer,
                        act_layer=act_layer
                    ),
                    name=f"up"
                )
            else:
                dec.add(
                    UpscaleModule(
                        is_first=(s==0),
                        in_channels=cfg.model.dec_channels[s],
                        out_channels=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        upscale_factor=cfg.model.upscale_factor[s],
                        n_frequencies=cfg.model.n_frequencies,
                        drop_path=dec_drop_path_[-1] \
                        if self.enable_upscale_drop_path else 0.0,
                        enable_absolute_pe=cfg.model.enable_absolute_pe,
                        enable_residual_attribute=self.enable_residual_attribute,
                        norm_layer=ln_layer,
                        act_layer=act_layer
                    ),
                    name=f"up"
                )

            if self.enable_residual_attribute:
                print('We use residual!')
                # head
                dec.add(
                    GaussianResModule(
                        is_first=(s==0),
                        dim=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        sh_degree=cfg.model.sh_degree,
                        enable_residual_attribute=self.enable_residual_attribute,
                        act_layer=act_layer
                    ),
                    name="head"
                )

                # mask module
                dec.add(
                    MaskResModule(
                        dim=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        temperature=cfg.model.temperature,
                        non_leaf_ratio=cfg.model.non_leaf_ratio[s] \
                        if s < len(cfg.model.dec_channels) - 1 else 1.0,
                        mask_sampling_type=cfg.model.mask_sampling_type,
                        act_layer=act_layer
                    ),
                    name="mask"
                )
            else:
                print('We do not use residual!')
                # mask module
                dec.add(
                    MaskModule(
                        dim=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        temperature=cfg.model.temperature,
                        non_leaf_ratio=cfg.model.non_leaf_ratio[s] \
                        if s < len(cfg.model.dec_channels) - 1 else 1.0,
                        mask_sampling_type=cfg.model.mask_sampling_type,
                        act_layer=act_layer
                    ),
                    name="mask"
                )

                # head
                dec.add(
                    GaussianModule(
                        is_first=(s==0),
                        dim=cfg.model.dec_channels[s+1] \
                        if s < len(cfg.model.dec_channels) - 1 else cfg.model.dec_channels[s],
                        sh_degree=cfg.model.sh_degree,
                        enable_residual_attribute=self.enable_residual_attribute,
                        act_layer=act_layer
                    ),
                    name="head"
                )

            self.dec.append(dec)

    def _union_gaussians(self, point_list):
        if self.cfg.model.use_mask:
            xyz_list = []
            attribute_list = []
            batch_index_list = []
            for i, point in enumerate(point_list):
                # append
                xyz_list.append(point.coord)
                attribute_list.append(point.attribute)
                batch_index_list.append(offset2batch(point.offset))

            # concatenate by group (i.e., batch)
            xyz, index = group_cat(xyz_list, batch_index_list, return_index=True)
            attribute = group_cat(attribute_list, batch_index_list)
            # offset = batch2offset(index)
        else:
            # TODO: Here we just use final layer points
            xyz = point_list[-1].coord
            attribute = point_list[-1].attribute

        sh, opacity, scaling, rotation = torch.split(attribute, [self.sh_dim,\
             self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)

        return xyz, sh, opacity, scaling, rotation

    def _union_res_gaussians(self, point_list, n_views, training=False, return_mask=False):
        xyz_list = []
        attribute_list = []
        batch_index_list = []
        current_lv_mask_list = []
        for i, point in enumerate(point_list):
            # select only leaf (NOTE: the last level is all leaf)
            if not training:
                if point.leaf is not None:
                    offset = point.leaf_offset
                    xyz = point.coord[point.leaf]
                    attribute = point.attribute[point.leaf]
                else:
                    offset = point.offset
                    xyz = point.coord
                    attribute = point.attribute
            else:
                if point.leaf is not None and i < len(point_list) - 1:
                    offset = point.leaf_offset
                    xyz = point.coord[point.leaf]
                    attribute = point.attribute[point.leaf]
                    if return_mask:
                        current_lv_mask = torch.zeros(
                            point.leaf_offset[-1], 
                            dtype=torch.bool, 
                            device=self.device
                        )
                else:
                    offset = point.offset
                    xyz = point.coord
                    attribute = point.attribute
                    if return_mask:
                        current_lv_mask = torch.ones(
                            point.offset[-1],
                            dtype=torch.bool,
                            device=self.device
                        )
            # append
            xyz_list.append(xyz)
            attribute_list.append(attribute)
            batch_index_list.append(offset2batch(offset))
            if training and return_mask:
                current_lv_mask_list.append(current_lv_mask)
        
        # concatenate by group (i.e., batch)
        xyz, index = group_cat(xyz_list, batch_index_list, return_index=True)
        attribute = group_cat(attribute_list, batch_index_list)
        offset = batch2offset(index)

        # # TODO: Here we just use final layer points
        # xyz = xyz_list[-1]
        # attribute = attribute_list[-1]

        if training and return_mask:
            current_lv_mask = group_cat(current_lv_mask_list, batch_index_list)
        else:
            current_lv_mask = None

        sh, opacity, scaling, rotation = torch.split(attribute, [self.sh_dim,\
             self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)

        return xyz, sh, opacity, scaling, rotation, current_lv_mask, offset

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'),dim=-1)
        grid = (grid + 0.5) / reso * 2 -1
        return grid.reshape(reso,reso,reso,3)*self.scene_size

    
    def build_feat_vol(self, src_inps, img_feats, n_views_sel, batch):

        h,w = src_inps.shape[-2:]
        src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)
        src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)
        

        img_wh = torch.tensor([w,h], device=self.device)
        point_img,_ = projection(self.volume_grid, src_w2cs, src_ixts) 
        point_img = (point_img+ 0.5)/img_wh*2 - 1.0

        # viewing direction
        rays = batch['tar_rays_down'][:,:n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1,*rays.shape[2:])
        feats_dir = torch.cat((rsh_cart_3(feats_dir[...,:3]),rsh_cart_3(feats_dir[...,3:6])),dim=-1)

        # query features
        img_feats =  torch.einsum('bchw->bhwc',img_feats)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw',img_feats)

        n_channel = img_feats.shape[1]
        feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        # img features
        feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)

        return feats_vol
    
    def _check_mask(self, mask):
        ratio = torch.sum(mask)/np.prod(mask.shape)
        if ratio < 1e-3: 
            mask = mask + torch.rand(mask.shape, device=self.device)>0.8
        elif  ratio > 0.5 and self.training: 
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device)>0.5
        return mask
            
    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):
        points = points[mask]
        n_points = points.shape[0]
        
        h,w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx,:n_views_sel].reshape(-1,3,3)
        src_w2cs = batch['tar_w2c'][idx,:n_views_sel].reshape(-1,4,4)
        
        img_wh = torch.tensor([w,h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5)/img_wh*2 - 1.0

        imgs_coarse = torch.cat((renderings['image'],renderings['acc_map'].unsqueeze(-1),renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)),dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel,-1,n_points).to(imgs_coarse)
        
        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1,n_points)).abs()
                    
        point_feats = torch.cat((feats_coarse[:,:-1],z_diff), dim=1)#[...,_mask]
        
        return point_feats, mask

    def ray_to_plucker(self, rays):
        origin, direction = rays[...,:3], rays[...,3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)
        
        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)
        
        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment),dim=-1)
    
    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5*self.scene_size/self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B,-1,K,-1).reshape(offset.shape) + offset*half_cell_size
        return centers
    
    def forward(self, batch, with_fine=False, return_buffer=True):
        
        B,N,H,W,C = batch['tar_rgb'].shape
        if self.training:
            n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        else:
            n_views_sel = self.cfg.n_views

        _inps =batch['tar_rgb'][:,:n_views_sel].reshape(B*n_views_sel,H,W,C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
        token_size = int(np.sqrt(H*W/img_feats.shape[-1]))
        img_feats = img_feats.reshape(*img_feats.shape[:2],H//token_size,W//token_size)

        # build 3D volume
        feat_vol = self.build_feat_vol(_inps, img_feats, n_views_sel, batch) # B n_views_sel C D H W
        
        # view embedding
        if self.cfg.model.view_embed_dim > 0:
            feat_vol = torch.cat((feat_vol, self.view_embed[:,:n_views_sel].expand(B,-1,-1,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)),dim=2)

        # decoding
        volume_feat_up = self.vol_decoder(feat_vol)

        # rendering
        _offset_coarse, _shs_coarse, _scaling_coarse, _rotation_coarse, _opacity_coarse = self.decoder.forward_coarse(volume_feat_up, self.opacity_shift, self.scaling_shift)

        # convert to local positions
        _centers_coarse = self.get_offseted_pt(_offset_coarse, self.K)
        _opacity_coarse_tmp = self.gs_render.opacity_activation(_opacity_coarse).squeeze(-1)
        masks =  _opacity_coarse_tmp > 0.005

        render_img_scale = batch.get('render_img_scale', 1.0)
        
        volume_feat_up = volume_feat_up.view(B,-1,volume_feat_up.shape[-1])
        _inps = _inps.reshape(B,n_views_sel,C,H,W).float()
        
        outputs,render_pkg = [],[]
        for i in range(B):
 
            znear, zfar = batch['near_far'][i]
            fovx,fovy = batch['fovx'][i], batch['fovy'][i]
            height, width = int(batch['meta']['tar_h'][i]*render_img_scale), int(batch['meta']['tar_w'][i]*render_img_scale)

            mask = masks[i].detach()

            _centers = _centers_coarse[i]
            if return_buffer:
                render_pkg.append((_centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i]))
            
            outputs_view = []
            tar_c2ws = batch['tar_c2w'][i]
            for j, c2w in enumerate(tar_c2ws):
                
                bg_color = batch['bg_color'][i,j]
                self.gs_render.set_bg_color(bg_color)
            
                cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                rays_d = batch['tar_rays'][i,j]
                
                # coarse
                frame = self.gs_render.render_img(cam, rays_d, _centers, _shs_coarse[i], _opacity_coarse[i],\
                        _scaling_coarse[i], _rotation_coarse[i], self.device)
                outputs_view.append(frame)
                
            rendering_coarse = {k: torch.stack([d[k] for d in outputs_view[:n_views_sel]]) for k in outputs_view[0]}

            # helper function to isolate fn from inputs not differentiated
            def get_grad_fn(centers, shs, opacity, scaling, rotation):
                # function to differentiate
                def fn(screenspace_point):
                    outputs_image=[]
                    tar_c2ws = batch['tar_c2w'][i]
                    for j, c2w in enumerate(tar_c2ws[:n_views_sel]):
                        bg_color = batch['bg_color'][i,j]
                        self.gs_render.set_bg_color(bg_color)
                        cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                        rays_d = batch['tar_rays'][i,j]
                        # coarse
                        frame = self.gs_render.render_img(cam, rays_d, centers, shs, opacity,\
                                scaling, rotation, self.device, screenspace_points=screenspace_point)
                        outputs_image.append(frame)
                    image_coarse = {k: torch.stack([d[k] for d in outputs_image]) for k in outputs_image[0]}
                    B_gt,V_gt,H_gt,W_gt = batch['tar_rgb'][:, :n_views_sel, :, :, :].shape[:-1]
                    image_co = image_coarse['image'].permute(1,0,2,3).reshape(H_gt,V_gt*W_gt,3).unsqueeze(0)
                    image_gt = batch['tar_rgb'][:, :n_views_sel, :, :, :].permute(0,2,1,3,4).reshape(B_gt,H_gt,V_gt*W_gt,3)
                    image_loss = ((image_co - image_gt)**2).mean()
                    return image_loss
                return fn

            if with_fine:
                # We do render and backward here to determine which point needs to be densified
                screenspace_point = torch.zeros((_centers.shape[0], 4), dtype=_centers.dtype,\
                            requires_grad=True, device=self.device) + 0
                # get the function to differentiate
                fn = get_grad_fn(_centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i])
                # gradients of image loss w.r.t screenspace point
                image_loss, grad = vjp(fn, screenspace_point)

                # point decoder fine, we delete with_fine here, just do coarse and fine at same time
                mask = self._check_mask(mask)
                point_grad = grad[mask]
                # We are using abs gradient here
                gradient_point = torch.norm(point_grad[:, 2:4], dim=-1, keepdim=True).squeeze()

                # We use threshold, top_p and top_k method here
                # # Threshold method
                # threshold = 0.00002
                # selected_pts_mask = torch.where(gradient_point >= 0.00002, True, False).squeeze()
                # # Top_p method
                # p_ratio = 0.5
                # k_num = int(gradient_point.shape[0] * p_ratio)
                # Top_k method
                if gradient_point.shape[0] < self.k_num:
                    selected_pts_mask = torch.where(gradient_point >= 0, True, False).squeeze()
                else:
                    topk_values, topk_indices = torch.topk(gradient_point, self.k_num, dim=0)
                    selected_pts_mask = torch.zeros_like(gradient_point, dtype=torch.bool)
                    selected_pts_mask[topk_indices] = True

                # # we densify all points here
                # selected_pts_mask = torch.ones_like(gradient_point, dtype=torch.bool)

                point_feats, mask = self.get_point_feats(i, _inps[i], rendering_coarse, n_views_sel, batch, _centers, mask)
                point_feats =  torch.einsum('lcb->blc', point_feats)
                volume_point_feat = volume_feat_up[i].unsqueeze(1).expand(-1,self.K,-1)[mask.view(-1,self.K)]
                _feats_fine, _shs_fine = self.decoder.forward_fine(volume_point_feat, point_feats)

                # This mask is from opacity(alpha) mask
                _shs_fine = _shs_fine.squeeze().view(-1,*_shs_coarse.shape[-2:]) + _shs_coarse[i][mask]
                ####### concatenate fine level features with coarse level features helps a lot on performance
                _features_fine = torch.concat((_feats_fine.squeeze(), volume_point_feat), dim=1)
                _centers_fine = _centers[mask]

                # We get fine points and densified points, use selected_pts_mask to determine if we need point-dec here
                _features_mask = _features_fine[selected_pts_mask]
                if self.enable_residual_attribute:
                    _centers_mask = _centers_fine[selected_pts_mask] * 2.0
                else:
                    _centers_mask = _centers_fine[selected_pts_mask]

                # initialize points
                _offset_mask = (torch.arange(1, 1 + 1) * _features_mask.shape[0]).long().to(self.device, non_blocking=True)
                if self.enable_residual_attribute:
                    data_dict = {
                        "grid_size": self.voxel_size, # 0.5 * self.voxel_size,
                        "coord": _centers_mask,
                        "feat": _features_mask,
                        "offset": _offset_mask,
                    }
                else:
                    data_dict = {
                        "grid_size": 0.5 * self.voxel_size,
                        "coord": _centers_mask,
                        "feat": _features_mask,
                        "offset": _offset_mask,
                    }

                point = Point(data_dict)
                point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
                point.sparsify()

                # # decoding
                out_points = []
                for dec in self.dec:
                    point = dec(point)
                    if self.enable_residual_attribute:
                        out_points.append(point)
                    else:
                        out_points.append(point.leaf_point)
                point_list = out_points

                # union all levels gaussians
                if self.enable_residual_attribute:
                    xyz_pt, shs_pt, opacity_pt, scaling_pt, rotation_pt, __, __ = self._union_res_gaussians(
                                        point_list=point_list, n_views=4, training=True, return_mask=False)
                    xyz_pt = xyz_pt / 2.0
                else:
                    xyz_pt, shs_pt, opacity_pt, scaling_pt, rotation_pt = self._union_gaussians(point_list=point_list)

                _centers = torch.cat([xyz_pt, _centers_coarse[i][mask][~selected_pts_mask]], dim=0)
                _shs = torch.cat([shs_pt.view(xyz_pt.shape[0], (self.sh_degree+1)**2, 3).float(), _shs_fine[~selected_pts_mask]], dim=0)
                _opacity = torch.cat([opacity_pt.float() + self.opacity_shift, _opacity_coarse[i][mask][~selected_pts_mask]], dim=0)
                _scaling = torch.cat([scaling_pt.float() + self.fine_scaling_shift, _scaling_coarse[i][mask][~selected_pts_mask]], dim=0)
                _rotation = torch.cat([rotation_pt.float(), _rotation_coarse[i][mask][~selected_pts_mask]], dim=0)

                if return_buffer:
                    render_pkg.append((_centers, _shs, _opacity, _scaling, _rotation, mask))

                for j,c2w in enumerate(tar_c2ws):
                    
                    bg_color = batch['bg_color'][i,j]
                    self.gs_render.set_bg_color(bg_color)
                
                    rays_d = batch['tar_rays'][i,j]
                    cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                    frame_fine = self.gs_render.render_img(cam, rays_d, _centers, _shs, _opacity, _scaling, _rotation, self.device, prex='_fine')
                    outputs_view[j].update(frame_fine)

            outputs.append({k: torch.cat([d[k] for d in outputs_view], dim=1) for k in outputs_view[0]})

        outputs = {k: torch.stack([d[k] for d in outputs]) for k in outputs[0]}
        if return_buffer:
            outputs.update({'render_pkg':render_pkg}) 
        return outputs