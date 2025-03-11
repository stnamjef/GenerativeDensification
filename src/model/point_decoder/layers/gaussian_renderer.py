#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(
    fovx,
    fovy,
    image_width,
    image_height,
    world_view_transform,
    full_proj_transform,
    camera_center,
    position,
    sh_coeff,
    opacity,
    scaling,
    rotation,
    screenspace_points,
    bg_color,
    sh_degree,
    override_color=None,
    autocast=False,
    debug=False
):
    """
    Render the scene.
    
    Background  tensor (bg_color) must be on GPU!
    """
    if sh_coeff is not None:
        assert sh_coeff.ndim == 3 and 3 * (sh_degree + 1)**2 == sh_coeff.size(-2) * sh_coeff.size(-1)

    # # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(position, dtype=position.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass

    # Set up rasterization configuration
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(image_height),
        image_width=int(image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = position
    means2D = screenspace_points

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = sh_coeff
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if autocast:  # disable autocat: rasterizer supports only fp32
        with torch.cuda.amp.autocast(enabled=False):
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D = means3D.float(),
                means2D = means2D.float(),
                shs = None if shs is None else shs.float(),
                colors_precomp = None if colors_precomp is None else colors_precomp.float(),
                opacities = opacity.float(),
                scales = scaling.float(),
                rotations = rotation.float(),
                cov3D_precomp = None
            )
    else:
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scaling,
            rotations = rotation,
            cov3D_precomp = None
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "visibility_filter" : radii > 0,
            "radii": radii}
