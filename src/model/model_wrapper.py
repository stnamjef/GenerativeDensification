from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json, os

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

from .ply_export import export_ply
from math import isqrt
from ..geometry.projection import get_fov, homogenize_points, project
from .decoder.cuda_splatting import get_projection_matrix
from torch.autograd.functional import vjp
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from .point_decoder.decoder import PointDecoder
from .types import Gaussians


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        pointdecoder_cfg,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

        # We register background for calculating mask gradient here, we assume 0.0 here
        self.register_buffer(
            "background_color",
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            persistent=False,
        )

        # Our Point Decoder module
        self.k_num = pointdecoder_cfg.k_num
        self.grid_size = pointdecoder_cfg.grid_size
        self.point_decoder = PointDecoder(
                            stride=pointdecoder_cfg.stride,
                            dec_depths=pointdecoder_cfg.dec_depths,
                            dec_channels=pointdecoder_cfg.dec_channels,
                            dec_num_head=pointdecoder_cfg.dec_num_head,
                            dec_patch_size=pointdecoder_cfg.dec_patch_size,
                            mlp_ratio=pointdecoder_cfg.mlp_ratio,
                            qkv_bias=pointdecoder_cfg.qkv_bias,
                            # global pooling
                            enable_ada_lnnorm=pointdecoder_cfg.enable_ada_lnnorm,
                            # upscale block
                            upscale_factor=pointdecoder_cfg.upscale_factor,
                            n_frequencies=pointdecoder_cfg.n_frequencies,
                            enable_absolute_pe=pointdecoder_cfg.enable_absolute_pe,
                            enable_upscale_drop_path=pointdecoder_cfg.enable_upscale_drop_path,
                            # mask block
                            non_leaf_ratio=pointdecoder_cfg.non_leaf_ratio,
        )
        # Fine details cross-attention module
        self.in_dim = 160
        cond_dim = 8
        self.sh_degree = pointdecoder_cfg.sh_degree
        self.sh_dim = (self.sh_degree + 1)**2*3  # sh_degree is 4
        self.mlp1 = nn.Sequential(
            nn.Linear(163, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, self.in_dim))
        self.norm = nn.LayerNorm(self.in_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=self.in_dim, num_heads=16, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        # # TODO: not using this after concat module
        self.mlp2 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, self.in_dim)) # + self.sh_dim))

        self.init(self.mlp1)
        self.init(self.mlp2)

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

    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):
        # Not using grad_mask here
        # points = points[mask]
        points = points.view(-1,3).unsqueeze(0).repeat(n_views_sel, 1, 1)
        n_points = points.shape[1]

        h,w = img_ref.shape[-2:]
        src_ixts = batch["context"]["intrinsics"][idx].reshape(-1,3,3)
        src_w2cs = batch["context"]["extrinsics"][idx].reshape(-1,4,4)
        src_ixts = src_ixts.unsqueeze(dim=1)
        src_w2cs = src_w2cs.unsqueeze(dim=1)

        img_wh = torch.tensor([w,h], device=self.device)
        point_xy, __, point_z = project(points, src_w2cs, src_ixts)
        point_xy = point_xy * 2 - 1.0

        imgs_coarse = torch.cat((renderings.color[idx],renderings.alpha[idx].unsqueeze(dim=1),renderings.depth[idx].unsqueeze(dim=1)), dim=1)
        imgs_coarse = torch.cat((img_ref, imgs_coarse),dim=1)
        ### padding_mode='border'
        feats_coarse = nn.functional.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False, padding_mode='border').view(n_views_sel,-1,n_points).to(imgs_coarse)

        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1, n_points)).abs()

        point_feats = torch.cat((feats_coarse[:,:-1],z_diff), dim=1)

        return point_feats.detach()

    def gradient_mask_calculation(self, batch, gaussians, target_gt):
        # helper function to isolate fn from inputs not differentiated
        def get_grad_fn(gaussian_means, shs, gaussian_opacities, gaussian_covariances, image_gt):
            # function to differentiate
            def fn(screenspace_point):
                v = image_gt.shape[1]
                for j in range(v):
                    settings = GaussianRasterizationSettings(
                        image_height=h,
                        image_width=w,
                        tanfovx=tan_fov_x[i*v+j].item(),
                        tanfovy=tan_fov_y[i*v+j].item(),
                        bg=self.background_color,
                        scale_modifier=1.0,
                        viewmatrix=view_matrix[i*v+j],
                        projmatrix=full_projection[i*v+j],
                        sh_degree=degree,
                        campos=extrinsics[i*v+j, :3, 3],
                        prefiltered=False,  # This matches the original usage.
                        debug=False,
                    )
                    rasterizer = GaussianRasterizer(settings)

                    row, col = torch.triu_indices(3, 3)

                    screenspace_point = screenspace_point.clone()
                    image_co, feature_map, alpha, depth, radii = rasterizer(
                        means3D=gaussian_means[i],
                        means2D=screenspace_point,
                        shs=shs[i] if use_sh else None,
                        colors_precomp=None if use_sh else shs[i, :, 0, :],
                        opacities=gaussian_opacities[i, ..., None],
                        cov3D_precomp=gaussian_covariances[i, :, row, col],
                    )

                    if j == 0:
                        image_loss = ((image_co - image_gt[i, j])**2).mean()
                    else:
                        image_loss = image_loss + ((image_co - image_gt[i, j])**2).mean()
                return image_loss
            return fn

        b, v, _, _ = batch["context"]["extrinsics"].shape 
        extrinsics = rearrange(batch["context"]["extrinsics"], "b v i j -> (b v) i j")
        intrinsics = rearrange(batch["context"]["intrinsics"], "b v i j -> (b v) i j")
        near = rearrange(batch["context"]["near"], "b v -> (b v)")
        far = rearrange(batch["context"]["far"], "b v -> (b v)")
        use_sh = True

        _, _, _, n = gaussians.harmonics.shape
        degree = isqrt(n) - 1

        _, _, _, h, w = batch["context"]["image"].shape

        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        tan_fov_x = (0.5 * fov_x).tan()
        tan_fov_y = (0.5 * fov_y).tan()

        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        full_projection = view_matrix @ projection_matrix

        all_images_grad = []

        for i in range(b):
            # We do render and backward here to determine which point needs to be densified
            screenspace_point = torch.zeros_like(gaussians.means[i], requires_grad=True)
            # get the function to differentiate
            fn = get_grad_fn(gaussians.means, gaussians.harmonics.permute(0, 1, 3, 2).contiguous(), gaussians.opacities, gaussians.covariances, target_gt)
            # gradients of image loss w.r.t screenspace point
            _ , grad = vjp(fn, screenspace_point)
            all_images_grad.append(grad)

        all_images_grad = torch.stack(all_images_grad)
        gradient_point = torch.norm(all_images_grad[:, :, 0:2], dim=-1, keepdim=True).squeeze(dim=-1)

        # Top_k 
        k_num = self.k_num
        assert h*w > k_num
        for i in range(int(gradient_point.shape[-1]/(h*w))):
            topk_values, topk_indices = torch.topk(gradient_point[:, i*h*w:(i+1)*h*w], k_num, dim=1)
            if i == 0:
                selected_pts_mask = torch.zeros_like(gradient_point[:, i*h*w:(i+1)*h*w], dtype=torch.bool)
                selected_pts_mask.scatter_(1, topk_indices, True)
            else:
                selected_pts_mask_index = torch.zeros_like(gradient_point[:, i*h*w:(i+1)*h*w], dtype=torch.bool)
                selected_pts_mask_index.scatter_(1, topk_indices, True)
                selected_pts_mask = torch.concat([selected_pts_mask, selected_pts_mask_index], dim=1)

        return selected_pts_mask, k_num * int(gradient_point.shape[-1]/(h*w))

    def rendering_fine(self, batch, gaussians, selected_gradient_mask, k_num, output_context):
        b, v, _, h, w = batch["context"]["image"].shape
        gaussians.means = rearrange(gaussians.means, "b (v h w) xyz -> b v (h w) xyz", v=v, h=h, w=w)
        gaussians.covariances = rearrange(gaussians.covariances, "b (v h w) i j -> b v (h w) i j", v=v, h=h, w=w)
        gaussians.harmonics = rearrange(gaussians.harmonics, "b (v h w) m n -> b v (h w) m n", v=v, h=h, w=w)
        gaussians.opacities = rearrange(gaussians.opacities, "b (v h w) -> b v (h w)", v=v, h=h, w=w)
        gaussians.features = rearrange(gaussians.features, "b (v h w) f -> b v (h w) f", v=v, h=h, w=w)
        selected_gradient_mask = rearrange(selected_gradient_mask, "b (v h w) -> b v (h w)", v=v, h=h, w=w)

        all_gaussians_fine_means = []
        all_gaussians_fine_covariances = []
        all_gaussians_fine_harmonics = []
        all_gaussians_fine_opacities = []
        for i_b in range(b):
            # We extract fine details features from cross-attention
            point_feats = self.get_point_feats(i_b, batch["context"]["image"][i_b], output_context, \
                v, batch, gaussians.means[i_b], torch.flatten(selected_gradient_mask[i_b]))
            point_feats =  torch.einsum('lcb->blc', point_feats)
            coarse_features = gaussians.features[i_b].view(-1, 163).contiguous()
            coarse_features = self.mlp1(coarse_features)
            coarse_features = self.norm(coarse_features.unsqueeze(1))
            feats_fine = self.cross_att(coarse_features, point_feats, point_feats, need_weights=False)[0]
            _feats_out = self.mlp2(feats_fine)
            _feats_fine = _feats_out
            _features_fine = _feats_fine[torch.flatten(selected_gradient_mask[i_b][:v])].squeeze()
            centers_offset = 0
            for i_v in range(v):
                _centers = gaussians.means[i_b,i_v,:,:][selected_gradient_mask[i_b,i_v,:]]
                _features = gaussians.features[i_b,i_v,:,:][selected_gradient_mask[i_b,i_v,:]]
                _features = self.mlp1(_features)
                _features = torch.concat((_features, _features_fine[centers_offset:(centers_offset+_centers.shape[0])].squeeze()), dim=1)
                centers_offset = _centers.shape[0]
                _offset = (torch.arange(1, 1 + 1) * _centers.shape[0]).long().to(self.device, non_blocking=True)
                data_dict = {
                    "grid_size": self.grid_size,
                    "coord": _centers,
                    "feat": _features,
                    "offset": _offset,
                }
                extrinsics = batch["context"]["extrinsics"][i_b,i_v,:,:]
                intrinsics = batch["context"]["intrinsics"][i_b,i_v,:,:]
                gaussians_dense = self.point_decoder(  
                    data_dict, 
                    batch,
                    extrinsics,
                    intrinsics,
                    (h,w))
                update_harmonics = gaussians.harmonics[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]
                if i_v == 0:
                    gaussians_fine_means = torch.cat([gaussians_dense.means.squeeze(), \
                                    gaussians.means[i_b,i_v,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_covariances = torch.cat([gaussians_dense.covariances.squeeze(), \
                                    gaussians.covariances[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_harmonics = torch.cat([gaussians_dense.harmonics.squeeze(), update_harmonics], dim=0)
                    gaussians_fine_opacities = torch.cat([gaussians_dense.opacities.squeeze(), \
                                    gaussians.opacities[i_b,i_v,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                else:
                    gaussians_fine_means = torch.cat([gaussians_fine_means, gaussians_dense.means.squeeze(), \
                                    gaussians.means[i_b,i_v,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_covariances = torch.cat([gaussians_fine_covariances, gaussians_dense.covariances.squeeze(), \
                                    gaussians.covariances[i_b,i_v,:,:,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
                    gaussians_fine_harmonics = torch.cat([gaussians_fine_harmonics, gaussians_dense.harmonics.squeeze(), \
                                    update_harmonics], dim=0)
                    gaussians_fine_opacities = torch.cat([gaussians_fine_opacities, gaussians_dense.opacities.squeeze(), \
                                    gaussians.opacities[i_b,i_v,:][~selected_gradient_mask[i_b,i_v,:]]], dim=0)
            
            all_gaussians_fine_means.append(gaussians_fine_means)
            all_gaussians_fine_covariances.append(gaussians_fine_covariances)
            all_gaussians_fine_harmonics.append(gaussians_fine_harmonics)
            all_gaussians_fine_opacities.append(gaussians_fine_opacities)

        all_gaussians_fine_means = torch.stack(all_gaussians_fine_means)
        all_gaussians_fine_covariances = torch.stack(all_gaussians_fine_covariances)
        all_gaussians_fine_harmonics = torch.stack(all_gaussians_fine_harmonics)
        all_gaussians_fine_opacities = torch.stack(all_gaussians_fine_opacities)

        gaussians_fine = Gaussians(
                all_gaussians_fine_means,
                all_gaussians_fine_covariances,
                all_gaussians_fine_harmonics,
                all_gaussians_fine_opacities,
        )
        output_fine = self.decoder.forward(
            gaussians_fine,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        
        ######### For visualization
        gaussians.means = gaussians.means.flatten(1,2)
        gaussians.covariances = gaussians.covariances.flatten(1,2)
        gaussians.harmonics = gaussians.harmonics.flatten(1,2)
        gaussians.opacities = gaussians.opacities.flatten(1,2)
        
        return output_fine, gaussians_fine

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians  = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        output_context = self.decoder.forward(
            gaussians,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        # Picking gradient_based point first, then do rendering on fine image
        selected_gradient_mask, k_num = self.gradient_mask_calculation(batch, gaussians, batch["context"]["image"])
        output_fine, __ = self.rendering_fine(batch, gaussians, selected_gradient_mask, k_num, output_context)

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        psnr_probabilistic_fine = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output_fine.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic_fine", psnr_probabilistic_fine.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            loss_fine = loss_fn.forward(output_fine, batch, gaussians, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            self.log(f"loss/{loss_fn.name}", loss_fine)
            total_loss = total_loss + loss + loss_fine
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):

        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )

        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        
        output_context = self.decoder.forward(
            gaussians,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        # Picking gradient_based point and then do rendering on fine image
        selected_gradient_mask, k_num = self.gradient_mask_calculation(batch, gaussians, batch["context"]["image"])
        output_fine, gaussians_fine = self.rendering_fine(batch, gaussians, selected_gradient_mask, k_num, output_context)
        images_prob_fine = output_fine.color[0]

        # Save images.
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], images_prob):
                save_image(color, path / scene / f"color/{index:0>6}_coarse.png")
            for index, color in zip(batch["target"]["index"][0], images_prob_fine):
                save_image(color, path / scene / f"color/{index:0>6}_fine.png")
            for index, color in zip(batch["target"]["index"][0], batch["target"]["image"].squeeze(dim=0)):
                save_image(color, path / scene / f"color/{index:0>6}_gt.png")

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            
            coarse = compute_psnr(rgb_gt, images_prob).mean().item()
            fine = compute_psnr(rgb_gt, images_prob_fine).mean().item()

            if f"psnr_coarse" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr_coarse"] = []
            if f"psnr_fine" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr_fine"] = []
            self.test_step_outputs[f"psnr_coarse"].append(coarse)
            self.test_step_outputs[f"psnr_fine"].append(fine)
                       
            if coarse > fine:
                rgb = images_prob
            else:
                rgb = images_prob_fine

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

        # save video
        if self.test_cfg.save_video: # and gap_num > 2.0:
            #### coarse image
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}_coarse.mp4",
            )
            #### fine image
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob_fine],
                path / "video" / f"{scene}_frame_{frame_str}_fine.mp4",
            )
            #### fine image
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in rgb_gt],
                path / "video" / f"{scene}_frame_{frame_str}_gt.mp4",
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        # render coarse image
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]
        
        # Picking gradient_based point first, then do rendering on fine image
        target_gt = batch["target"]["image"]

        output_context = self.decoder.forward(
            gaussians_softmax,
            batch["context"]["extrinsics"],
            batch["context"]["intrinsics"],
            batch["context"]["near"],
            batch["context"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )

        selected_gradient_mask, k_num = self.gradient_mask_calculation(batch, gaussians_softmax, batch["context"]["image"])
        output_softmax_fine, __ = self.rendering_fine(batch, gaussians_softmax, selected_gradient_mask, k_num, output_context)
        rgb_softmax_fine = output_softmax_fine.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)
        for tag, rgb in zip(
            ("val",), (rgb_softmax_fine,)
        ):
            psnr_fine = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_fine_{tag}", psnr_fine)
            lpips_fine = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_fine_{tag}", lpips_fine)
            ssim_fine = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_fine_{tag}", ssim_fine)


        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        comparison_fine = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax_fine), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison_fine",
            [prep_image(add_border(comparison_fine))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)


    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
