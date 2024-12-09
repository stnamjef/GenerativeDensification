import os

gpu_id = 0
name = 'release'
###### original version
ckpt_path = f'checkpoints/epoch=49.ckpt'

for n_views in [4]:
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_GSO_{n_views}_views.json ' \
        f'infer.dataset.dataset_name=GSO infer.dataset.data_root=../LaRa/dataset/google_scanned_objects infer.eval_depth=[0.005,0.01,0.02] ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_GSO_{n_views}_views infer.dataset.n_group={n_views} '
    os.system(cmd) 

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_gobjeverse_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_gobjaverse_{n_views}_views infer.dataset.n_group={n_views} '
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_teddybear_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/Co3D/co3d_teddybear.hdf5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_co3d_teddybear infer.dataset.n_group={n_views} '
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_hydrant_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/Co3D/co3d_hydrant.hdf5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs/image_vis/{name}_co3d_hydrant infer.dataset.n_group={n_views} '
    os.system(cmd)

###### residual version
ckpt_path = f'checkpoints/epoch=49_residual.ckpt'

for n_views in [4]:
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_GSO_{n_views}_views.json ' \
        f'infer.dataset.dataset_name=GSO infer.dataset.data_root=../LaRa/dataset/google_scanned_objects infer.eval_depth=[0.005,0.01,0.02] ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs_res/image_vis/{name}_GSO_{n_views}_views infer.dataset.n_group={n_views} ' \
        f'model.enable_residual_attribute=True'
    os.system(cmd) 

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_gobjeverse_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/gobjaverse/gobjaverse.h5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs_res/image_vis/{name}_gobjaverse_{n_views}_views infer.dataset.n_group={n_views} ' \
        f'model.enable_residual_attribute=True'
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_teddybear_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/Co3D/co3d_teddybear.hdf5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs_res/image_vis/{name}_co3d_teddybear infer.dataset.n_group={n_views} ' \
        f'model.enable_residual_attribute=True'
    os.system(cmd)
    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python evaluation.py configs/infer.yaml n_views={n_views} infer.eval_novel_view_only=True ' \
        f'infer.ckpt_path={ckpt_path} infer.metric_path=outputs/metrics/{name}_co3d_hydrant_{n_views}_views.json  ' \
        f'infer.dataset.dataset_name=gobjeverse infer.dataset.data_root=../LaRa/dataset/Co3D/co3d_hydrant.hdf5 ' \
        f'infer.video_frames=0 infer.save_mesh=False ' \
        f'infer.save_folder=outputs_res/image_vis/{name}_co3d_hydrant infer.dataset.n_group={n_views} ' \
        f'model.enable_residual_attribute=True'
    os.system(cmd)