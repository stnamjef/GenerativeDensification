import math
import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.functional import vjp
from torch.autograd.functional import (
    _as_tuple, _grad_preprocess, 
    _grad_postprocess, _check_requires_grad, 
    _validate_v, _autograd_grad, 
    _fill_in_zeros, _tuple_postprocess)
from .autoencoder import AutoEncoder
from .layers.gaussian_renderer import render
from .utils.misc import offset2batch, batch2offset


def custom_vjp(func, inputs, v=None, create_output_graph=True, create_result_graph=False, strict=False):
    r"""Compute the dot product between a vector ``v`` and the Jacobian of the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector
            Jacobian product is computed.  Must be the same size as the output
            of ``func``. This argument is optional when the output of ``func``
            contains a single element and (if it is not provided) will be set
            as a Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vjp (tuple of Tensors or Tensor): result of the dot product with
            the same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def exp_reducer(x):
        ...     return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...     return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_output_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_result_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        else:
            if len(outputs) != 1 or outputs[0].nelement() != 1:
                raise RuntimeError(
                    "The vector v can only be None if the "
                    "user-provided function returns "
                    "a single Tensor with a single element."
                )

    enable_grad = True if create_result_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_result_graph)
        vjp = _fill_in_zeros(grad_res, inputs, strict, create_result_graph, "back")

    # Cleanup objects and return them to the user
    outputs = _grad_postprocess(outputs, create_output_graph)
    vjp = _grad_postprocess(vjp, create_result_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        vjp, is_inputs_tuple
    )


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


def mse_to_psnr(mse):
    return 0 if mse == 0 else -10. * math.log(mse) / math.log(10.)


class Model:
    def __init__(
        self, 
        rank, 
        sh_degree,
        scale_activation_scale,
        scale_activation_shift,
        enable_mask_module,
        model_kwargs,
        rendering_kwargs
    ):
        self.rank = rank
        self.sh_degree = sh_degree
        self.scale_activation_scale = scale_activation_scale
        self.scale_activation_shift = scale_activation_shift
        self.enable_mask_module = enable_mask_module
        self.rendering_kwargs = rendering_kwargs

        self.autoencoder = DDP(
            AutoEncoder(**model_kwargs).to(rank),
            device_ids=[rank,]
        )

        # slices to access each attribute
        self.num_sh = 3 * (self.sh_degree + 1)**2
        self.slice_sh = slice(0, self.num_sh)
        self.slice_opacity = slice(self.num_sh, self.num_sh+1)
        self.slice_scale = slice(self.num_sh+1, self.num_sh+1+3)
        self.slice_rotation = slice(self.num_sh+1+3, self.num_sh+1+3+4)

        self.opacity_activation = torch.sigmoid
        self.scale_activation = lambda x: torch.exp(scale_activation_scale * torch.tanh(x) + scale_activation_shift)
        self.rotation_activation = lambda x: F.normalize(x, dim=-1)

        # background color
        if rendering_kwargs["white_background"]:
            self.background = torch.tensor([1., 1., 1.], device=rank)
        else:
            self.background = torch.tensor([0., 0., 0.], device=rank)
    
    def forward(self, data_dict, camera, images_gt):
        point_list = self.autoencoder(data_dict)
        self._render_and_get_loss(point_list, camera, images_gt)
    
    def render(self, data_dict, camera):
        point_list = self.autoencoder(data_dict)
        return self._render_only(point_list, camera)
    
    def _activation(self, attribute):
        activated = torch.empty(attribute.shape, device=attribute.device, dtype=attribute.dtype)
        activated[..., self.slice_sh] = attribute[..., self.slice_sh]
        activated[..., self.slice_opacity] = self.opacity_activation(attribute[..., self.slice_opacity])
        activated[..., self.slice_scale] = self.scale_activation(attribute[..., self.slice_scale])
        activated[..., self.slice_rotation] = self.rotation_activation(attribute[..., self.slice_rotation])
        return activated
    
    def _update_screenspace_point(self, point_list, n_views):
        # update (add) screenspace point
        for point in point_list:
            # NOTE: check if requires_grad=True needed?
            point.update({
                "screenspace_point": torch.zeros(
                    size=(point.offset[-1], n_views, 3),
                    dtype=point.coord.dtype,
                    device=self.rank
                )
            })
        return point_list
    
    def _union_gaussians(self, point_list, training=False):
        xyz_list = []
        attribute_list = []
        batch_index_list = []
        screenspace_point_list = []
        current_lv_mask_list = []
        for i, point in enumerate(point_list):
            # select only leaf (NOTE: the last level is all leaf)
            if not training:
                if point.leaf is not None:
                    offset = point.leaf_offset
                    xyz = point.coord[point.leaf]
                    attribute = point.attribute[point.leaf]
                    screenspace_point = point.screenspace_point[point.leaf]
                else:
                    offset = point.offset
                    xyz = point.coord
                    attribute = point.attribute
                    screenspace_point = point.screenspace_point
            else:
                if point.leaf is not None and i < len(point_list) - 1:
                    offset = point.leaf_offset
                    xyz = point.coord[point.leaf]
                    attribute = point.attribute[point.leaf]
                    screenspace_point = point.screenspace_point[point.leaf]
                    current_lv_mask = torch.zeros(
                        point.leaf_offset[-1], 
                        dtype=torch.bool, 
                        device=self.rank
                    )
                else:
                    offset = point.offset
                    xyz = point.coord
                    attribute = point.attribute
                    screenspace_point = point.screenspace_point
                    current_lv_mask = torch.ones(
                        point.offset[-1],
                        dtype=torch.bool,
                        device=self.rank
                    )
            # append
            xyz_list.append(xyz)
            attribute_list.append(attribute)
            batch_index_list.append(offset2batch(offset))
            screenspace_point_list.append(screenspace_point)
            if training:
                current_lv_mask_list.append(current_lv_mask)
        
        # concatenate by group (i.e., batch)
        xyz, index = group_cat(xyz_list, batch_index_list, return_index=True)
        attribute = group_cat(attribute_list, batch_index_list)
        screenspace_point = group_cat(screenspace_point_list, batch_index_list)
        offset = batch2offset(index)

        if training:
            current_lv_mask = group_cat(current_lv_mask_list, batch_index_list)
        else:
            current_lv_mask = None
        
        return xyz, self._activation(attribute), screenspace_point, current_lv_mask, offset
    
    def _render_batch(self, xyz, attribute, screenspace_point, offset, camera):
        # number of batches and views
        n_batches, n_views = camera.FoVx.shape

        # loop through all batches
        begin, images = 0, []
        for i in range(n_batches):
            end = offset[i]
            _xyz = xyz[begin:end]
            _attribute = attribute[begin:end]
            _screenspace_point = screenspace_point[begin:end]
            begin = end
            # loop through all views
            for j in range(n_views):
                render_pkg = render(
                    camera.FoVx[i, j],
                    camera.FoVy[i, j], 
                    camera.image_width[i, j], 
                    camera.image_height[i, j], 
                    camera.world_view_transform[i, j], 
                    camera.full_proj_transform[i, j], 
                    camera.camera_center[i, j], 
                    _xyz,
                    _attribute[..., self.slice_sh].view(len(_xyz), (self.sh_degree+1)**2, 3),
                    _attribute[..., self.slice_opacity],
                    _attribute[..., self.slice_scale],
                    _attribute[..., self.slice_rotation],
                    _screenspace_point[:, j, :],
                    self.background,
                    self.sh_degree,
                    autocast=self.rendering_kwargs["autocast"]
                )
                images.append(render_pkg["render"])
        images = torch.stack(images, dim=0)
        images = torch.unflatten(images, dim=0, sizes=(n_batches, n_views))

        return images
    
    @torch.no_grad()
    def _render_only(self, point_list, camera):
        assert camera.FoVx.ndim == 2
        assert not self.autoencoder.training

        # update (add) screenspace point
        point_list = self._update_screenspace_point(point_list)

        # loop through all levels
        images_all = []
        for lv in range(len(point_list)):
            # union up to the current level and render
            xyz, attribute, screenspace_point, _, offset = self._union_gaussians(point_list[:lv+1])

            # render a batch of images
            images = self._render_batch(xyz, attribute, screenspace_point, offset, camera)

            # append
            images_all.append(images)

        return images_all
    
    def _render_and_get_loss(self, point_list, camera, images_gt):
        assert camera.FoVx.ndim == 2
        assert self.autoencoder.training

        # helper function to isolate fn from inputs not differentiated
        def get_grad_fn(xyz, attribute, offset):
            # function to differentiate
            def fn(screenspace_point):
                images = self._render_batch(xyz, attribute, screenspace_point, offset, camera)
                image_loss = F.mse_loss(images, images_gt)
                return image_loss
            return fn
        
        def render_up_to_this_lv(point_list, create_output_graph=False):
            # union up to the current level and render
            with torch.set_grad_enabled(create_output_graph):
                xyz, attribute, screenspace_point, current_lv_mask, offset = self._union_gaussians(point_list[:lv+1], True)

            # get the function to differentiate
            fn = get_grad_fn(xyz, attribute, offset)

            if self.enable_mask_module:
                # current lv point
                current_lv_point = point_list[-1]

                # gradients of image loss w.r.t screenspace point
                image_loss, grad = custom_vjp(fn, screenspace_point, None, create_output_graph)

                # slice only the current lv (filter all the previous levels)
                grad = grad[current_lv_mask]

                # average gradients over views: (N, V, 3) -> (N, 3)
                avg_grad = torch.mean(grad, dim=1)

                # norm of gradients: (N, 1)
                grad_norm = torch.norm(avg_grad, dim=-1, keepdim=True)

                # sum of gradients of each batch: (N, 1)
                offset_padded = F.pad(current_lv_point.offset, (1, 0), "constant", 0)
                grad_norm_sum = torch_scatter.segment_csr(
                    src=grad_norm.to(torch.float32),  # to avoid torch_scatter bug when using float16
                    indptr=offset_padded,
                    reduce="sum"
                )
                grad_norm_sum_repeated = torch_scatter.gather_csr(
                    src=grad_norm_sum,
                    indptr=offset_padded
                )

                # normalize grad_norm to [0, 1]
                grad_norm_normalized = grad_norm / grad_norm_sum_repeated

                # gradient loss: (N, 1)
                grad_loss = F.mse_loss(current_lv_point.prob, grad_norm_normalized, reduction="none")
                
                # reduce each batch and average all
                grad_loss = torch_scatter.segment_csr(
                    src=grad_loss,
                    indptr=offset_padded,
                    reduce="mean"
                ).mean()

                return image_loss, grad_loss
            
            return fn(screenspace_point), None

        # update (add) screenspace point
        point_list = self._update_screenspace_point(point_list, camera.FoVx.shape[1])

        # loop through all levels
        image_loss_all = []
        grad_loss_all = []
        for lv in range(len(point_list)):
            # render image and calculate loss
            image_loss, grad_loss = render_up_to_this_lv(
                point_list=point_list[:lv+1], 
                create_output_graph=(lv==(len(point_list) - 1))
            )
            
            # calculate addtional gradient loss
            import pdb; pdb.set_trace()


    # def _render(self, point_list, camera, images_gt):
    #     assert camera.FoVx.ndim == 2

    #     def render_up_to_this_lv(point_list):          
    #         # union of Gaussians up to the current level
    #         xyz, attribute, screenspace_point, offset = self._union_gaussians(point_list)

    #         if self.autoencoder.training:
    #             # v for vector-jacobian product
    #             v = torch.tensor(1.0, device=self.rank)

    #             # get the function to differentiate
    #             fn = get_grad_fn(xyz, attribute, offset)

    #             # gradients of MSE loss w.r.t screenspace point
    #             (images, image_loss), grad = vjp(fn, screenspace_point, (None, v))

    #             # average gradients over views: (N, V, 3) -> (N, 3)
    #             avg_grad = torch.mean(grad, dim=1)

    #             # norm of gradients: (N, 1)
    #             grad_norm = torch.norm(avg_grad, dim=-1, keepdim=True)

    #             # sum of gradients of each batch: (N, 1)
    #             offset_padded = F.pad(offset, (1, 0), "constant", 0)
    #             grad_norm_sum = torch_scatter.segment_csr(
    #                 src=grad_norm.to(torch.float32),  # to prevent torch_scatter bug on float16
    #                 indptr=offset_padded,
    #                 reduce="sum"
    #             )
    #             grad_norm_sum_repeated = torch_scatter.gather_csr(
    #                 src=grad_norm_sum,
    #                 indptr=offset_padded
    #             )

    #             # normalize grad_norm to [0, 1]
    #             grad_norm_normalized = grad_norm / grad_norm_sum_repeated

    #         return images, image_loss, grad_norm_normalized

    #     # update (add) screenspace point
    #     for point in point_list:
    #         # NOTE: check if requires_grad=True needed?
    #         point.update({
    #             "screenspace_point": torch.zeros(
    #                 size=(point.offset[-1], camera.FoVx.shape[1], 3),
    #                 dtype=point.coord.dtype,
    #                 device=self.rank
    #             )
    #         })

    #     # loop through all levels and render images
    #     image_all = []
    #     image_loss_all = []
    #     grad_norm_all = []
    #     for lv in range(len(point_list)):
    #         # union up to the current level and render
    #         if lv < len(point_list) - 1:
    #             with torch.no_grad():
    #                 images, image_loss, grad_norm = render_up_to_this_lv(point_list[:lv+1])
    #         else:
    #             images, image_loss, grad_norm = render_up_to_this_lv(point_list[:lv+1])
    #         # append all images
    #         image_all.append(images)
    #         image_loss_all.append(image_loss)
    #         grad_norm_all.append(grad_norm)
    
    # def _render(self, point_list, camera, images_gt):

    #     def get_render_fn(xyz, attribute, offset, grad_fn=False):
    #         # normal rendering function
    #         def fn(screenspace_point)

    #         # function to differentiate
    #         def grad_fn(screenspace_point):
    #             # render a batch of images
    #             images = self._render_batch(xyz, attribute, screenspace_point, offset)

    #             # image loss
    #             loss = F.mse_loss(images, images_gt)

    #             return images, loss
            
    #         return grad_fn if grad_fn else fn
        
    #     def render_up_to_this_lv(point_list):          
    #         # union of Gaussians up to the current level
    #         xyz, attribute, screenspace_point, offset = union_gaussians(point_list)

    #         # v for vector-jacobian product
    #         v = torch.tensor(1.0, device=self.rank)

    #         # get the function to differentiate
    #         fn = get_grad_fn(xyz, attribute, offset)

    #         # gradients of MSE loss w.r.t screenspace point
    #         (images, image_loss), grad = vjp(fn, screenspace_point, (None, v))

    #         # average gradients over views: (N, V, 3) -> (N, 3)
    #         avg_grad = torch.mean(grad, dim=1)

    #         # norm of gradients: (N, 1)
    #         grad_norm = torch.norm(avg_grad, dim=-1, keepdim=True)

    #         # sum of gradients of each batch: (N, 1)
    #         offset_padded = F.pad(offset, (1, 0), "constant", 0)
    #         grad_norm_sum = torch_scatter.segment_csr(
    #             src=grad_norm.to(torch.float32),  # to prevent torch_scatter bug on float16
    #             indptr=offset_padded,
    #             reduce="sum"
    #         )
    #         grad_norm_sum_repeated = torch_scatter.gather_csr(
    #             src=grad_norm_sum,
    #             indptr=offset_padded
    #         )

    #         # normalize grad_norm to [0, 1]
    #         grad_norm_normalized = grad_norm / grad_norm_sum_repeated

    #         return images, image_loss, grad_norm_normalized
        
    #     # update point (add screenspace point)
    #     for point in point_list:
    #         # check if requires_grad=True needed?
    #         point.update({
    #             "screenspace_point": torch.zeros(
    #                 size=(point.offset[-1], n_views, 3),
    #                 dtype=point.coord.dtype,
    #                 device=self.rank
    #             )
    #         })

    #     # loop through all levels
    #     image_all = []
    #     image_loss_all = []
    #     grad_norm_all = []
    #     for lv in range(len(point_list)):
    #         # union up to the current level and render
    #         if lv < len(point_list) - 1:
    #             with torch.no_grad():
    #                 images, image_loss, grad_norm = render_up_to_this_lv(point_list[:lv+1])
    #         else:
    #             images, image_loss, grad_norm = render_up_to_this_lv(point_list[:lv+1])
    #         # append all images
    #         image_all.append(images)
    #         image_loss_all.append(image_loss)
    #         grad_norm_all.append(grad_norm)
        
    #     return image_all, image_loss_all, grad_norm_all
    
    @torch.no_grad()
    def attribute_statistics(self, point_list):
        # container
        stats_all = {"num_points": [], "opacity": [], "scale": []}

        # loop through all levels
        for point in point_list:
            # get leaf attribute
            if point.leaf is not None:
                offset = point.leaf_offset
                attribute = point.attribute[point.leaf]
            else:
                offset = point.offset
                attribute = point.attribute
            attribute = self._activation(attribute)
            # average over batches
            assert len(attribute) == offset[-1]
            attribute_mean = torch_scatter.segment_csr(
                src=attribute.to(torch.float32),
                indptr=F.pad(offset, (1, 0), "constant", 0),
                reduce="mean"
            ).mean(dim=0)
            # append
            stats_all["num_points"].append(offset[0].item())
            stats_all["opacity"].append(attribute_mean[self.slice_opacity].mean().item())
            stats_all["scale"].append(attribute_mean[self.slice_scale].mean().item())
        
        # reformat
        stats_reformatted = {}
        for key, stats in stats_all.items():
            stats_reformatted.update({f"{key} (mean; l{i})": stat for i, stat in enumerate(stats)})

        return stats_reformatted

    def train(self):
        self.autoencoder.train(mode=True)
    
    def eval(self):
        self.autoencoder.train(mode=False)

    def total_params(self):
        return sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
    
    def parameters(self):
        return self.autoencoder.parameters()
    
    def clip_grad_norm(self, max_norm):
        original_grad_norm = clip_grad_norm_(self.autoencoder.parameters(), max_norm)
        return original_grad_norm
 
    def state_dict(self):
        return self.autoencoder.state_dict()
     
    def load_state_dict(self, state_dict, strict):
        self.autoencoder.load_state_dict(state_dict, strict)


def setup_model(rank, cfg):
    if rank == 0:
        print("Building model...")

    if not cfg.model.enable_mask_module:
        cfg.model.non_leaf_ratio = [1.0 for _ in range(len(cfg.model.dec_channels)-1)]

    # Model kwargs
    model_kwargs = {
        "in_channels": cfg.model.in_channels,
        "order": cfg.model.order,
        "stride": cfg.model.stride,
        "enc_depths": cfg.model.enc_depths,
        "enc_channels": cfg.model.enc_channels,
        "enc_num_head": cfg.model.enc_num_head,
        "enc_patch_size": cfg.model.enc_patch_size,
        "dec_depths": cfg.model.dec_depths,
        "dec_channels": cfg.model.dec_channels,
        "dec_num_head": cfg.model.dec_num_head,
        "dec_patch_size": cfg.model.dec_patch_size,
        "mlp_ratio": cfg.model.mlp_ratio,
        "qkv_bias": cfg.model.qkv_bias,
        "qk_scale": cfg.model.qk_scale,
        "attn_drop": cfg.model.attn_drop,
        "proj_drop": cfg.model.proj_drop,
        "drop_path": cfg.model.drop_path,
        "pre_norm": cfg.model.pre_norm,
        "shuffle_orders": cfg.model.shuffle_orders,
        "enable_rpe": cfg.model.enable_rpe,
        "enable_flash": cfg.model.enable_flash,
        "upcast_attention": cfg.model.upcast_attention,
        "upcast_softmax": cfg.model.upcast_softmax,
        "cls_mode": cfg.model.cls_mode,
        "pdnorm_bn": cfg.model.pdnorm_bn,
        "pdnorm_ln": cfg.model.pdnorm_ln,
        "pdnorm_decouple": cfg.model.pdnorm_decouple,
        "pdnorm_adaptive": cfg.model.pdnorm_adaptive,
        "pdnorm_affine": cfg.model.pdnorm_affine,
        # affine
        "bnnorm_affine": cfg.model.bnnorm_affine,
        "lnnorm_affine": cfg.model.lnnorm_affine,
        # global
        "enable_ada_lnnorm": cfg.model.enable_ada_lnnorm,
        # upscale block
        "upscale_factor": cfg.model.upscale_factor,
        "n_frequencies": cfg.model.n_frequencies,
        "enable_absolute_pe": cfg.model.enable_absolute_pe,
        "enable_upscale_drop_path": cfg.model.enable_upscale_drop_path,
        # mask block
        "non_leaf_ratio": cfg.model.non_leaf_ratio,
        # gaussian head
        "sh_degree": cfg.model.sh_degree,
        "enable_residual_attribute": cfg.model.enable_residual_attribute
    }

    rendering_kwargs = {
        "white_background": cfg.dataset.white_background,
        "autocast": cfg.train.autocast
    }

    # Build model
    if "dev" in cfg.model.name:
        raise NotImplementedError
    else:
        model = Model(
            rank, 
            cfg.model.sh_degree,
            cfg.model.scale_activation_scale,
            cfg.model.scale_activation_shift,
            cfg.model.enable_mask_module, 
            model_kwargs, 
            rendering_kwargs
        )

    if rank == 0:
        total_params = model.total_params()
        print(model.autoencoder)
        print(f"#Params: {total_params:,}, Size: {(total_params * 4)//(1024**2):,}MB")

    # Load params if available
    if cfg.model.resume is not None:
        ckpt = torch.load(cfg.model.resume, map_location={"cuda:0": f"cuda:{rank}"})
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Rank{rank}: Params loaded")

    return model
