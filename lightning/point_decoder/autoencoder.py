"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath
from typing import Callable, Tuple
from torch_geometric.utils import cumsum as pyg_cumsum
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import softmax as pyg_softmax
import pytorch_lightning as L

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .point_prompt_training import PDNorm
from .utils.misc import offset2bincount, offset2batch
from .utils.structure import Point
from .utils.modules import PointModule, PointSequential
from .layers.normalization import AdaLayerNorm

# from simple_knn._C import distCUDA2


def positional_encoding(x, base_freq):
    fx = torch.flatten(base_freq[None, :, None] * x[:, None, :], -2, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


class RPE(L.LightningModule):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(L.LightningModule):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
    
    def forward(self, point: Point):
        if not self.cpe[0].training:
            self.cpe[0].train(True)
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            # NOTE: MODIFIED
            grid_size=point.grid_size * self.stride,
            # NOTE: MODIFIED
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


class GlobalPooling(PointModule):
    def forward(self, point: Point):

        # prepend zero at the front
        offset_padded = F.pad(point.offset, (1, 0), "constant", 0)

        # average over each batch
        global_feat = torch_scatter.segment_csr(
            src=point.feat,
            indptr=offset_padded,
            reduce="mean"
        )

        # update point
        point.update({"global_feat": global_feat})

        return point


@torch.no_grad()
def top_k(x, ratio, batch):
    assert ratio > 0.0 and ratio < 1.0

    # number of points in each batch
    num_nodes = pyg_scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

    # number of points to be selected
    k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)
    
    # sort the whole x (do not consider batch, for now)
    x, x_perm = torch.sort(x.view(-1), descending=True)

    # now we sort batch, such that x_original[x_perm[batch_perm]]
    # is the same as the tensor sorted within each batch
    batch = batch[x_perm]
    batch, batch_perm = torch.sort(batch, descending=False, stable=True)

    # element indices within each batch; [0, 1, 2, 0, 1] if batch == [0, 0, 0, 1, 1]
    arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    ptr = pyg_cumsum(num_nodes)
    batched_arange = arange - ptr[batch]
    mask_sorted = batched_arange < k[batch]

    # each element indicates whether the corresponding x is non_leaf or not
    mask_unsorted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    mask_unsorted[x_perm[batch_perm[mask_sorted]]] = True

    return mask_unsorted, torch.cumsum(k, dim=0)


@torch.no_grad()
def top_p(x, ratio, offset):
    assert ratio > 0.0 and ratio < 1.0

    def group_cumsum(x, offset):
        # total number of points
        N = offset[-1]

        # offset padded with zero in front
        offset_padded = F.pad(offset, (1, 0), "constant", 0)
        
        # create sparse tensor:
        # ex) offset = [3, 5]
        # [[1, 0, 0, 0, 0]
        #  [1, 1, 0, 0, 0]
        #  [1, 1, 1, 0, 0]
        #  [0, 0, 0, 1, 0]
        #  [0, 0, 0, 1, 1]]
        indices = torch.cat([
            torch.tril_indices(end-begin, end-begin, device=x.device) + begin
            for begin, end in zip(offset_padded[:-1], offset_padded[1:])
        ], dim=-1)
        sparse_one_tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=torch.ones(indices.size(1), dtype=torch.float32, device=x.device),
            size=(N, N)
        )

        # cumsum by sparse matmul: sparse mm does not support f16
        with torch.cuda.amp.autocast(enabled=False):
            y = torch.mm(sparse_one_tensor, x[:, None].to(torch.float32))
        y = y.to(x.dtype).flatten()

        return y
    
    # offset to batch indices
    batch = offset2batch(offset)

    # sort the whole x (do not consider batch, for now)
    x, x_perm = torch.sort(x.view(-1), descending=True)

    # now we sort batch, such that x_original[x_perm[batch_perm]]
    # is the same as the tensor sorted within each batch
    batch = batch[x_perm]
    batch, batch_perm = torch.sort(batch, descending=False, stable=True)

    # cumsum the (sorted) probabilities within each batch
    x_cumsum = group_cumsum(x[batch_perm], offset)

    # (sorted) mask, where True indicates the non_leaf
    # in other words, True == (cumulative prob <= ratio)
    mask_sorted = x_cumsum <= ratio

    # each element indicates whether the corresponding x is non_leaf or not
    mask_unsorted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    mask_unsorted[x_perm[batch_perm[mask_sorted]]] = True

    # offset of non_leaf (mask == True)
    new_offset = torch_scatter.segment_csr(
        src=mask_unsorted.to(offset.dtype),
        indptr=F.pad(offset, (1, 0), "constant", 0),
        reduce="sum"
    ).cumsum(0)

    return mask_unsorted, new_offset


class MaskModule(PointModule):
    def __init__(
        self,
        dim: int,
        temperature: float,
        non_leaf_ratio: float,
        mask_sampling_type: str,
        act_layer: Callable[..., nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        assert non_leaf_ratio <= 1.0 and non_leaf_ratio > 0.0
        assert mask_sampling_type in ["topk", "topp"]

        self.dim = dim
        self.temperature = temperature
        self.non_leaf_ratio = non_leaf_ratio
        self.mask_sampling_type = mask_sampling_type

        if non_leaf_ratio < 1.0:
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                act_layer(),
                nn.Linear(dim, 1),
            )
    
    def _softmax(self, raw_prob, offset):
        prob = pyg_softmax(
            src=raw_prob.to(torch.float32) / self.temperature,
            ptr=F.pad(offset, (1, 0), "constant", 0),
            dim=0
        )
        return prob
        
    def forward(self, point: Point):
        if self.non_leaf_ratio < 1.0:
            # input feature: (N, C)
            feat = point.feat

            # probability to be densified
            raw_prob = self.net(feat)
            prob = torch.sigmoid(raw_prob)

            # boolean mask indicating non_leaf nodes and new offset
            if self.mask_sampling_type == "topk":
                non_leaf, non_leaf_offset = top_k(
                    x=prob, 
                    ratio=self.non_leaf_ratio, 
                    batch=offset2batch(point.offset)
                )
            else:
                non_leaf, non_leaf_offset = top_p(
                    x=prob,
                    ratio=self.non_leaf_ratio,
                    offset=point.offset
                )
            assert torch.sum(non_leaf) == non_leaf_offset[-1]
            leaf = ~non_leaf
            leaf_offset = point.offset - non_leaf_offset

            # straight through estimator: actual indexing will be done in SerializationModule
            # point.feat = (feat * non_leaf[:, None] - feat * prob).detach() + feat * prob
            point.feat = (feat - feat * prob).detach() + feat * prob

            # reset point
            point = Point(
                coord=point.coord[non_leaf],
                feat=point.feat[non_leaf],
                global_feat=point.global_feat,
                offset=non_leaf_offset,
                grid_size=point.grid_size,
                leaf_point=Point(
                    coord=point.coord[leaf],
                    feat=point.feat[leaf],
                    offset=leaf_offset,
                    grid_size=point.grid_size
                )
            )
        else:
            point = Point(
                coord=point.coord,
                feat=point.feat,
                global_feat=point.global_feat,
                offset=point.offset,
                grid_size=point.grid_size,
                leaf_point=Point(
                    coord=point.coord,
                    feat=point.feat,
                    offset=point.offset,
                    grid_size=point.grid_size
                )
            )
    
        return point

class MaskResModule(PointModule):
    def __init__(
        self,
        dim: int,
        temperature: float,
        non_leaf_ratio: float,
        mask_sampling_type: str,
        act_layer: Callable[..., nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        assert non_leaf_ratio <= 1.0 and non_leaf_ratio > 0.0
        assert mask_sampling_type in ["topk", "topp"]

        self.dim = dim
        self.temperature = temperature
        self.non_leaf_ratio = non_leaf_ratio
        self.mask_sampling_type = mask_sampling_type

        if non_leaf_ratio < 1.0:
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                act_layer(),
                nn.Linear(dim, 1),
            )
    
    def _softmax(self, raw_prob, offset):
        prob = pyg_softmax(
            src=raw_prob.to(torch.float32) / self.temperature,
            ptr=F.pad(offset, (1, 0), "constant", 0),
            dim=0
        )
        return prob
        
    def forward(self, point: Point):
        dict_to_update = {
            "raw_prob": None,
            "prob": None,
            "non_leaf": None,
            "non_leaf_offset": None,
            "leaf": None,
            "leaf_offset": None,
        }

        if self.non_leaf_ratio < 1.0:
            # input feature: (N, C)
            feat = point.feat

            # probability to be densified
            raw_prob = self.net(feat)
            prob = self._softmax(raw_prob, point.offset)

            # boolean mask indicating non_leaf nodes and new offset
            if self.mask_sampling_type == "topk":
                non_leaf, non_leaf_offset = top_k(
                    x=prob, 
                    ratio=self.non_leaf_ratio, 
                    batch=offset2batch(point.offset)
                )
            else:
                non_leaf, non_leaf_offset = top_p(
                    x=prob,
                    ratio=self.non_leaf_ratio,
                    offset=point.offset
                )
            assert torch.sum(non_leaf) == non_leaf_offset[-1]

            # straight through estimator: actual indexing will be done in SerializationModule
            point.feat = (feat * non_leaf[:, None] - feat * prob).detach() + feat * prob

            dict_to_update = {
                "raw_prob": raw_prob,
                "prob": prob,
                "non_leaf": non_leaf,
                "non_leaf_offset": non_leaf_offset,
                "leaf": ~non_leaf,
                "leaf_offset": point.offset - non_leaf_offset
            }

        # update point
        point.update(dict_to_update)
    
        return point

def positional_encoding(f, x):
    fx = torch.flatten(f[None, :, None] * x[:, None, :], -2, -1)
    return torch.cat([torch.sin(fx), torch.cos(fx)], dim=-1)


class UpscaleModule(PointModule):
    def __init__(
        self,
        is_first: bool,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        n_frequencies: int = 10,
        drop_path: float = 0.0,
        enable_absolute_pe: bool = False,
        enable_residual_attribute: bool = False,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert upscale_factor > 1, "upscale_factor > 1"

        self.is_first = is_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        self.n_frequencies = n_frequencies
        self.enable_absolute_pe = enable_absolute_pe
        self.enable_residual_attribute = enable_residual_attribute
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.in_norm = PointSequential(norm_layer(in_channels))

        self.delta_x = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            act_layer(),
            nn.Linear(in_channels, 3 * upscale_factor)
        )

        self.skip = nn.Linear(in_channels, out_channels)

        if n_frequencies > 0:
            _in_channels = in_channels + 3 * 2 * n_frequencies
        else:
            _in_channels = in_channels
        
        self.delta_f = nn.Sequential(
            nn.LayerNorm(_in_channels, elementwise_affine=False),
            nn.Linear(_in_channels, in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels)
        )
        self.out_norm = PointSequential(norm_layer(out_channels))

        if n_frequencies > 0:
            self.register_buffer("frequencies", 2.0**torch.arange(n_frequencies))

    def forward(self, point: Point):
        # pre-normalization
        point = self.in_norm(point)

        # preserve coord and feat
        in_x = point.coord
        in_f = point.feat

        # total number of points
        N = len(in_x)

        # repeat x and f: (N*S, 3 | C)
        temp_offset = torch.arange(
            N + 1,
            dtype=torch.int64,
            device=in_x.device
        ) * self.upscale_factor
        skip_x = torch_scatter.gather_csr(in_x, temp_offset)
        skip_f = torch_scatter.gather_csr(in_f, temp_offset)

        # delta coordinates: (N*S, 3)
        delta_x = self.delta_x(in_f).reshape(N*self.upscale_factor, 3)
        delta_x = 0.5 * point.grid_size * torch.tanh(delta_x)

        # delta features: (N*S, C)
        if self.n_frequencies > 0:
            if self.enable_absolute_pe:
                _delta_x = positional_encoding(self.frequencies, skip_x+delta_x)
            else:
                _delta_x = positional_encoding(self.frequencies, delta_x)
        else:
            _delta_x = delta_x
        delta_f = self.delta_f(torch.cat([_delta_x, skip_f], dim=-1))

        # add delta to x and f
        out_x = skip_x + delta_x
        out_f = self.skip(skip_f) + self.drop_path(delta_f)

        # reset point coordinates and features
        # NOTE: no need to replace sparse_feat
        # this will be done in SeiralizationModule
        point.coord = out_x
        point.feat = out_f
        point.offset = point.offset * self.upscale_factor
        # if self.enable_residual_attribute and not self.is_first:
        #     point.attribute = torch_scatter.gather_csr(point.attribute, temp_offset)

        # out normalization
        point = self.out_norm(point)

        return point
    
class UpscaleResModule(PointModule):
    def __init__(
        self,
        is_first: bool,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        n_frequencies: int = 10,
        drop_path: float = 0.0,
        enable_absolute_pe: bool = False,
        enable_residual_attribute: bool = False,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ):
        super().__init__()
        assert upscale_factor > 1, "upscale_factor > 1"

        self.is_first = is_first
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        self.n_frequencies = n_frequencies
        self.enable_absolute_pe = enable_absolute_pe
        self.enable_residual_attribute = enable_residual_attribute
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.in_norm = PointSequential(norm_layer(in_channels))

        self.delta_x = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            act_layer(),
            nn.Linear(in_channels, 3 * upscale_factor)
        )

        self.skip = nn.Linear(in_channels, out_channels)

        if n_frequencies > 0:
            _in_channels = in_channels + 3 * 2 * n_frequencies
        else:
            _in_channels = in_channels
        
        self.delta_f = nn.Sequential(
            nn.LayerNorm(_in_channels, elementwise_affine=False),
            nn.Linear(_in_channels, in_channels),
            act_layer(),
            nn.Linear(in_channels, out_channels)
        )
        self.out_norm = PointSequential(norm_layer(out_channels))

        if n_frequencies > 0:
            self.register_buffer("frequencies", 2.0**torch.arange(n_frequencies))

    def forward(self, point: Point):
        # pre-normalization
        point = self.in_norm(point)

        # preserve coord and feat
        in_x = point.coord
        in_f = point.feat

        # total number of points
        N = len(in_x)

        # repeat x and f: (N*S, 3 | C)
        temp_offset = torch.arange(
            N + 1,
            dtype=torch.int64,
            device=in_x.device
        ) * self.upscale_factor
        skip_x = torch_scatter.gather_csr(in_x, temp_offset)
        skip_f = torch_scatter.gather_csr(in_f, temp_offset)

        # delta coordinates: (N*S, 3)
        delta_x = self.delta_x(in_f).reshape(N*self.upscale_factor, 3)
        delta_x = 0.5 * point.grid_size * torch.tanh(delta_x)

        # delta features: (N*S, C)
        if self.n_frequencies > 0:
            if self.enable_absolute_pe:
                _delta_x = positional_encoding(self.frequencies, skip_x+delta_x)
            else:
                _delta_x = positional_encoding(self.frequencies, delta_x)
        else:
            _delta_x = delta_x
        delta_f = self.delta_f(torch.cat([_delta_x, skip_f], dim=-1))

        # add delta to x and f
        out_x = skip_x + delta_x
        out_f = self.skip(skip_f) + self.drop_path(delta_f)

        # reset point coordinates and features
        # NOTE: no need to replace sparse_feat
        # this will be done in SeiralizationModule
        point.coord = out_x
        point.feat = out_f
        point.offset = point.offset * self.upscale_factor
        if self.enable_residual_attribute and not self.is_first:
            point.attribute = torch_scatter.gather_csr(point.attribute, temp_offset)

        # out normalization
        point = self.out_norm(point)

        return point


class SerializationModule(PointModule):
    def __init__(
        self,
        stride: int = 2,
        order: Tuple[str] = ("z", "z-trans"),
        shuffle_orders: bool = True,
        enable_residual_attribute: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.order = order
        self.shuffle_orders = shuffle_orders
        self.enable_residual_attribute = enable_residual_attribute

    def forward(self, point: Point):
        # point attributes
        point_dict = Dict(
            coord=point.coord,
            feat=point.feat,
            global_feat=point.global_feat,
            offset=point.offset,
            grid_size=point.grid_size / self.stride
        )

        # if self.enable_residual_attribute:
        #     point_dict.update({
        #         "attribute": point.attribute[point.non_leaf] \
        #         if point.non_leaf is not None else point.attribute
        #     })
        
        # create a new Point
        point = Point(point_dict)

        # serialize and sparsify
        point.serialization(
            order=self.order,
            shuffle_orders=self.shuffle_orders
        )
        point.sparsify()

        return point

class SerializationResModule(PointModule):
    def __init__(
        self,
        stride: int = 2,
        order: Tuple[str] = ("z", "z-trans"),
        shuffle_orders: bool = True,
        enable_residual_attribute: bool = False,
    ):
        super().__init__()

        self.stride = stride
        self.order = order
        self.shuffle_orders = shuffle_orders
        self.enable_residual_attribute = enable_residual_attribute

    def forward(self, point: Point):
        assert "non_leaf" in point.keys()

        # select only non-leaf node
        if point.non_leaf is not None:
            coord = point.coord[point.non_leaf]
            feat = point.feat[point.non_leaf]
            offset = point.non_leaf_offset
        else:
            coord = point.coord
            feat = point.feat
            offset = point.offset
        
        # point attributes
        point_dict = Dict(
            coord=coord,
            feat=feat,
            global_feat=point.global_feat,
            offset=offset,
            grid_size=point.grid_size / self.stride
        )

        if self.enable_residual_attribute:
            point_dict.update({
                "attribute": point.attribute[point.non_leaf] \
                if point.non_leaf is not None else point.attribute
            })
        
        # create a new Point
        point = Point(point_dict)

        # serialize and sparsify
        point.serialization(
            order=self.order,
            shuffle_orders=self.shuffle_orders
        )
        point.sparsify()

        return point

class GaussianModule(PointModule):
    def __init__(
        self,
        is_first: bool,
        dim: int,
        sh_degree: int = 0,
        enable_residual_attribute: bool = False,
        act_layer: Callable[..., nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        assert sh_degree <= 3, "sh_degree must be less than or equal to 3"

        self.dim = dim
        self.sh_degree = sh_degree
        self.num_sh = 3 * (sh_degree + 1)**2
        self.is_first = is_first
        self.enable_residual_attribute = enable_residual_attribute

        self.feat2attr = nn.Sequential(
            nn.Linear(dim, dim),
            act_layer(),
            # sh + opacity + scale + rotation
            nn.Linear(dim, self.num_sh + 1 + 3 + 4)
        )

        # slices to access each attribute
        self.slice_sh = slice(0, self.num_sh)
        self.slice_opacity = slice(self.num_sh, self.num_sh+1)
        self.slice_scale = slice(self.num_sh+1, self.num_sh+1+3)
        self.slice_rotation = slice(self.num_sh+1+3, self.num_sh+1+3+4)
    
    def forward(self, point: Point):
        # decode features into Gaussian attributes
        attribute = self.feat2attr(point.leaf_point.feat)
        # # residual connection
        # if self.enable_residual_attribute and not self.is_first:
        #     attribute = point.attribute + attribute
        # update the previous lv attribute to the current lv
        point.leaf_point.update({"attribute": attribute})
        return point

class GaussianResModule(PointModule):
    def __init__(
        self,
        is_first: bool,
        dim: int,
        sh_degree: int = 0,
        enable_residual_attribute: bool = False,
        act_layer: Callable[..., nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        assert sh_degree <= 3, "sh_degree must be less than or equal to 3"

        self.dim = dim
        self.sh_degree = sh_degree
        self.num_sh = 3 * (sh_degree + 1)**2
        self.is_first = is_first
        self.enable_residual_attribute = enable_residual_attribute

        self.feat2attr = nn.Sequential(
            nn.Linear(dim, dim),
            act_layer(),
            # sh + opacity + scale + rotation
            nn.Linear(dim, self.num_sh + 1 + 3 + 4)
        )

        # slices to access each attribute
        self.slice_sh = slice(0, self.num_sh)
        self.slice_opacity = slice(self.num_sh, self.num_sh+1)
        self.slice_scale = slice(self.num_sh+1, self.num_sh+1+3)
        self.slice_rotation = slice(self.num_sh+1+3, self.num_sh+1+3+4)
    
    def forward(self, point: Point):
        # decode features into Gaussian attributes
        attribute = self.feat2attr(point.feat)
        # residual connection
        # import pdb; pdb.set_trace()
        if self.enable_residual_attribute and not self.is_first:
            attribute = point.attribute + attribute
        # update the previous lv attribute to the current lv
        point.update({"attribute": attribute})
        return point


class AutoEncoder(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 6, 2, 2, 2),
        dec_channels=(512, 256, 128, 64, 32),
        dec_num_head=(2, 4, 8, 16, 32),
        dec_patch_size=(48, 48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
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
        bnnorm_affine=True,
        lnnorm_affine=True,
        # global pooling
        enable_ada_lnnorm=True,
        # upscale block
        upscale_factor=(2, 2, 2, 2, 2),
        n_frequencies=10,
        enable_absolute_pe=False,
        enable_upscale_drop_path=False,
        # mask block
        temperature=1.0,
        non_leaf_ratio=(0.9, 0.9, 0.9, 0.9),
        mask_sampling_type="topp",
        # gaussian head
        sh_degree=0,
        enable_residual_attribute=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert len(dec_depths) == len(dec_channels) \
            == len(dec_num_head) == len(dec_patch_size) \
            == len(upscale_factor) == len(non_leaf_ratio) + 1

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

        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[:s+1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
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
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        if enable_ada_lnnorm:
            ln_layer = partial(
                AdaLayerNorm,
                w_shape=enc_channels[-1]
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
                        order=order,
                        shuffle_orders=shuffle_orders,
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

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # encoding
        point = self.embedding(point)
        point = self.enc(point)

        # decoding
        out_points = []
        for dec in self.dec:
            point = dec(point)
            out_points.append(point)
        
        return out_points

