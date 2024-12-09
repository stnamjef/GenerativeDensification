import numpy as np
from glob import glob
import random
import torch
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R
import os
from PIL import Image
import json

import h5py

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

class ShapenetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(ShapenetDataset, self).__init__()
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.split = cfg.split
        self.img_size = np.array(cfg.img_size)

        self._load()

        # TODO: read shapenet dataset here
        # self.metas = h5py.File(self.data_root, 'r')
        # scenes_name = np.array(sorted(self.metas.keys()))
        self.scenes_name = self.data
        print('len', len(self.scenes_name))
        
        # if 'splits' in scenes_name:
        #     self.scenes_name = self.metas['splits']['test'][:].astype(str) #self.metas['splits'][self.split]
        # else:
        #     i_test = np.arange(len(scenes_name))[::10][:cfg.n_scenes]
        #     i_train = np.array([i for i in np.arange(len(scenes_name)) if
        #                     (i not in i_test)])[:cfg.n_scenes]
        #     self.scenes_name = scenes_name[i_train] if self.split=='train' else scenes_name[i_test]
            
        # self.b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.n_group = cfg.n_group

    def _load(self):
        # for all categories
        # directories = []
        # for id in self.category_ids:
        #     dir_path = os.path.join(self.path, id, "directories.txt")
        #     if not os.path.exists(dir_path):
        #         scan_shapenet(id, self.path)
        #     with open(dir_path) as f:
        #         directories += [d for d in f.read().split("\n") if d]

        directories = []
        dir_path = os.path.join(self.data_root, "directories_H100.txt")
        if not os.path.exists(dir_path):
            scan_shapenet(self.data_root)
        with open(dir_path) as f:
            directories += [d for d in f.read().split("\n") if d]

        # split into train (90%) and validation (10%) set
        if self.split == "train":
            begin, end = 0, int(len(directories) * 0.9)
        else:
            begin, end = int(len(directories) * 0.9), len(directories)
        print(f"Data loaded ({self.split}): {end - begin:,}/{len(directories):,}")
        
        self.data = directories[begin:end]
        random.Random(2024).shuffle(self.data)

    def __getitem__(self, index):

        # scene_name = self.scenes_name[index]
        # scene_info = self.metas[scene_name]

        scene_info = self.scenes_name[index]
        # print('scene_info', scene_info)
        scene_name = scene_info

        # if self.split=='train' and self.n_group > 1:
        #     src_view_id = [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        #     view_id = src_view_id + [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        # elif self.n_group == 1:
        #     src_view_id = [scene_info['groups'][f'groups_4_{i}'][0] for i in range(1)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        # else:
        #     src_view_id = [scene_info['groups'][f'groups_{self.n_group}_{i}'][0] for i in range(self.n_group)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        
        # TODO: Modify view sampling
        if self.split=='train' and self.n_group > 1:
            # For training
            numbers = list(range(150))
            src_view_id = random.sample(numbers, self.n_group)
            view_id = src_view_id + random.sample([num for num in numbers if num not in src_view_id], self.n_group)
        else:
            # For validation
            numbers = list(range(150))
            src_view_id = random.sample(numbers, self.n_group)
            view_id = src_view_id + random.sample([num for num in numbers if num not in src_view_id], self.n_group)
        tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts, fov = self.read_views(scene_info, view_id, scene_name)

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        tar_c2ws = transform_mats @ tar_c2ws.copy()
 
        ret = {'fovx': fov, 
               'fovy': fov,
               }
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'tar_msk': tar_msks,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors
                    })
        
        if self.cfg.load_normal:
            tar_nrms = tar_nrms @ transform_mats[0,:3,:3].T
            ret.update({'tar_nrm': tar_nrms.transpose(1,0,2,3).reshape(H,len(view_id)*W,3)})
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene_name, 'tar_view': view_id, 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret
    
    def read_views(self, scene, src_views, scene_name):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, msks, normals = [], [], [], [], [], []
        for i, idx in enumerate(src_ids):
            
            if self.split!='train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])

            bg_colors.append(bg_color)
            
            img, normal, mask = self.read_image(scene, idx, bg_color, scene_name)
            imgs.append(img)
            ixt, ext, w2c, fov = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
            normals.append(normal)
        return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts), fov

    def read_cam(self, scene, view_idx):

        with open(os.path.join(scene, "transforms.json")) as f:
            json_file = json.load(f)
            fov = json_file["camera_angle_x"]
            frames = json_file["frames"]

        b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        c2w = np.array(frames[view_idx]['transform_matrix'])
        c2w = c2w @ b2c
        c2w = np.array(c2w, dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        fov = np.array(fov, dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)

        return ixt, c2w, w2c, fov

    def read_image(self, scene, view_idx, bg_color, scene_name):

        # print('view', view_idx)
        # print('name', scene_name)
        
        img = np.array(Image.open(os.path.join(scene, f'{view_idx:03d}.png')))
        # img = np.array(scene[f'image_{view_idx}'])

        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)

        if self.cfg.load_normal:

            normal = np.array(scene[f'normal_{view_idx}'])
            normal = normal.astype(np.float32) / 255. * 2 - 1.0
            return img, normal, mask

        return img, None, mask


    def __len__(self):
        return len(self.scenes_name)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

def stratified_sampling(n_indices, n_bins):
    indices = np.arange(n_indices)
    bins = np.array_split(indices, n_bins)
    sampled_indices = np.array([np.random.choice(bin) for bin in bins])
    np.random.shuffle(sampled_indices)
    return sampled_indices

def scan_shapenet(path):
    missing = 0
    subpaths = []
    dir_to_scan = os.scandir(path)
    for f in tqdm(dir_to_scan, desc="Scanning dataset dir..."):
        if f.is_dir():
            subpath = os.path.abspath(f.path)
            subpaths.append(subpath)
    
    # write paths as a .txt file
    with open(os.path.join(path, "directories_H100.txt"), "w") as f:
        for subpath in subpaths:
            f.write(f"{subpath}\n")
    
    print(f"Sanning done: {len(subpaths):,}/{len(subpaths)+missing:,} ({missing:,} is missing)")
    print(f"File saved at {os.path.join(path, 'directories_H100.txt')}")