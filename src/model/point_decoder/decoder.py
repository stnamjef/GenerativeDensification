from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

# from ....geometry.projection import get_world_rays
from ...misc.sh_rotation import rotate_sh
from ..encoder.common.gaussians import build_covariance

try:
    import flash_attn
except ImportError:
    flash_attn = None

from functools import partial
from ..point_decoder.utils.structure import Point
from ..point_decoder.utils.modules import PointModule, PointSequential
from ..point_decoder.autoencoder import *
from ..point_decoder.utils.misc import offset2batch, batch2offset
from ..point_decoder.point_prompt_training import PDNorm
from typing import Callable, Tuple

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

@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]

class PointDecoder(PointModule):
    def __init__(
        self,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2), # (2, 2),
        dec_depths=(2, 2, 2), # (2, 2, 2),
        dec_channels=(320, 256, 128), # (320, 256, 128),
        dec_num_head=(40, 32, 16), # (40, 16, 32),
        dec_patch_size=(48, 48, 48), # (48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=False, #True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        # affine
        bnnorm_affine=False,
        lnnorm_affine=False,
        # global pooling
        enable_ada_lnnorm=True,
        # upscale block
        upscale_factor=(2, 2, 2), # (1, 2, 2),
        n_frequencies=15,
        enable_absolute_pe=False,
        enable_upscale_drop_path=True,
        # mask block
        temperature=1.0,
        non_leaf_ratio=(0.5, 0.8, 0.8), # (1.0, 1.0),
        mask_sampling_type="topk",
        # gaussian head
        sh_degree=4,
        enable_residual_attribute=True,
    ):
        super(PointDecoder, self).__init__()
        self.order = [order] if isinstance(order, str) else order
        # self.order = order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.sh_degree = sh_degree
        self.sh_dim = (self.sh_degree + 1)**2*3
        self.scaling_dim, self.rotation_dim = 3, 4
        self.opacity_dim = 1
        self.gaussian_scale_min = 0.5
        self.gaussian_scale_max = 15.0
        self.d_sh = (self.sh_degree + 1)**2
        
        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        # activation layers
        act_layer = nn.GELU

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(
                nn.BatchNorm1d, 
                eps=1e-3, 
                momentum=0.01, 
                track_running_stats=False,
                affine=bnnorm_affine
            )
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = partial(
                nn.LayerNorm, 
                elementwise_affine=lnnorm_affine
            )

        # decoder
        dec_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
        ][::-1]
        self.dec = nn.ModuleList([])
        # self.head = nn.ModuleList([])
        for s in range(len(dec_channels)):
            dec = PointSequential()

            if s > 0:
                dec.add(
                    SerializationModule(
                        stride=stride[s-1],
                        # order=("z", "z-trans"),
                        shuffle_orders=self.shuffle_orders,
                        enable_residual_attribute=enable_residual_attribute
                        ),
                    name="serialization"
                )
            else:
                if enable_ada_lnnorm:
                    dec.add(
                        GlobalPooling(),
                        name="global"
                    )

            # attention block
            dec_drop_path_ = dec_drop_path[
                sum(dec_depths[:s]) : sum(dec_depths[:s+1])
            ]
            for i in range(dec_depths[s]):
                dec.add(
                    Block(
                        channels=dec_channels[s],
                        num_heads=dec_num_head[s],
                        patch_size=dec_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=dec_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            
            # upscale module
            dec.add(
                UpscaleModule(
                    is_first=(s==0),
                    in_channels=dec_channels[s],
                    out_channels=dec_channels[s+1] \
                    if s < len(dec_channels) - 1 else dec_channels[s],
                    upscale_factor=upscale_factor[s],
                    n_frequencies=n_frequencies,
                    drop_path=dec_drop_path_[-1] \
                    if enable_upscale_drop_path else 0.0,
                    enable_absolute_pe=enable_absolute_pe,
                    enable_residual_attribute=enable_residual_attribute,
                    norm_layer=ln_layer,
                    act_layer=act_layer
                ),
                name=f"up"
            )

            # head
            dec.add(
                GaussianModule(
                    is_first=(s==0),
                    dim=dec_channels[s+1] \
                    if s < len(dec_channels) - 1 else dec_channels[s],
                    sh_degree=sh_degree,
                    enable_residual_attribute=enable_residual_attribute,
                    act_layer=act_layer
                ),
                name="head"
            )

            # mask module
            dec.add(
                MaskModule(
                    dim=dec_channels[s+1] \
                    if s < len(dec_channels) - 1 else dec_channels[s],
                    temperature=temperature,
                    non_leaf_ratio=non_leaf_ratio[s] \
                    if s < len(dec_channels) - 1 else 1.0,
                    mask_sampling_type=mask_sampling_type,
                    act_layer=act_layer
                ),
                name="mask"
            )

            self.dec.append(dec)
            
    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def _union_gaussians(self, point_list, training=False, return_mask=False):
        # import pdb; pdb.set_trace()
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

        if training and return_mask:
            current_lv_mask = group_cat(current_lv_mask_list, batch_index_list)
        else:
            current_lv_mask = None

        sh, opacity, scaling, rotation = torch.split(attribute, [self.sh_dim,\
             self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)

        return xyz, attribute, offset

    def forward(self, 
            data_dict, 
            batch,
            extrinsics,
            intrinsics,
            image_shape,
            eps: float = 1e-8):

        # import pdb; pdb.set_trace()
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # decoding
        out_points = []
        for dec in self.dec:
            point = dec(point)
            out_points.append(point)
        
        # # TODO: Here we just use final layer points
        # xyz_list = []
        # attribute_list = []
        # batch_index_list = []
        # for i, point in enumerate(out_points):
        #     offset = point.offset
        #     xyz = point.coord
        #     attribute = point.attribute
        #     # append
        #     xyz_list.append(xyz)
        #     attribute_list.append(attribute)
        #     batch_index_list.append(offset2batch(offset))
        # xyz = xyz_list[-1]
        # attribute = attribute_list[-1]


        # TODO: Here we use mask-learning model and trying to prune redundant points
        point_list = out_points
        # union all levels gaussians
        xyz, attribute, offset = self._union_gaussians(
            point_list=point_list, training=True, return_mask=False)

        B = len(offset)
        N = offset[0]
        xyz = xyz.view(B,-1,xyz.shape[-1])
        attribute = attribute.view(B, 1, -1, 1, 1, attribute.shape[-1])
        dec_sh, dec_opacities, dec_scales, dec_rotations = torch.split(attribute, [self.sh_dim,\
             self.opacity_dim, self.scaling_dim, self.rotation_dim], dim=-1)
        dec_opacities = dec_opacities.squeeze(dim=-1)

        # We do not need to compute means again, just get means from point-decoder
        means = xyz.view(B, 1, -1, 1, 1, xyz.shape[-1])

        # Map scale features to valid scale range.
        scale_min = self.gaussian_scale_min
        scale_max = self.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * dec_scales.sigmoid()
        pixel_size = 1 / torch.tensor(image_shape, dtype=torch.float32, device=self.device)
        origins = extrinsics[..., :-1, -1]
        intrinsics = intrinsics.view(B, 1, 1, 1, 1, 3, 3)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        depths = torch.norm(means-origins, dim=-1)
        # print('depths:', depths.max(), depths.min())
        scales = scales * depths[..., None] * multiplier[..., None]

        # Normalize the quaternion features to yield a valid quaternion. eps same as MVSplat
        rotations = dec_rotations / (dec_rotations.norm(dim=-1, keepdim=True) + eps)

        # Apply sigmoid to get valid colors.
        sh = rearrange(dec_sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*dec_opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        extrinsics = extrinsics.view(B, 1, 1, 1, 1, 4, 4)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Broadcast opacities, we use sigmoid activation to get opacity factor from point decoder
        opacities = dec_opacities.sigmoid()

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )