import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.general_utils import AttrDict
from PIL import Image
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from scene.clip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')



if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    # parser.add_argument("--clip", action='store_true')
    # parser.add_argument("--lisa", action='store_true')
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = AttrDict(json.load(f)[args.scene])
    args = AttrDict(args.__dict__)
    args.update(cfg)
    # if 'rgb' in args.feature_gs_source:
    #     rgb_decode = True
    # else:
    #     rgb_decode = False
    # if 'depth' in args.feature_gs_source:
    #     depth_decode = True
    # else:
    #     depth_decode = False
    gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
    gaussian.load_ply(args.gs_source)
    if args.feature_gs_source:
        gaussian.load_feature_params(args.feature_gs_source)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
    colmap_cameras = None
    render_cameras = None
    #clip
    clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    clip.set_positives(args.clip_text_prompts)
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        scene = CamScene(args.colmap_dir, h=h, w=w, eval=True)
        colmap_train_cameras = scene.cameras
        colmap_eval_cameras = scene.eval_cameras
        new_colmap_eval_cameras = []
        eval_images = os.listdir(os.path.join(args.colmap_dir, 'test_mask'))
        for i in range(len(colmap_eval_cameras)):
            test_index = colmap_eval_cameras[i].image_name.split('.')[0].split('_')[-1]
            if test_index in eval_images:
                new_colmap_eval_cameras.append(colmap_eval_cameras[i])
        colmap_eval_cameras = new_colmap_eval_cameras
        # img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        # imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        # imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    # point_instance_embeddings = None
    # clip_semantic_embeddings = None
    
    rendered_feature_pca_dict = None
    instance_feature_pca_dict = None
    clip_feature_pca_dict = None
    # point_mask_map_save_path = os.path.join(args.save_path, 'point', 'mask_maps')
    # point_mask_save_path = os.path.join(args.save_path, 'point', 'masks')
    # point_mask_object_save_path = os.path.join(args.save_path, 'point', 'object')
    rgb_save_path = os.path.join(args.save_path, 'rgbs')
    clip_relevancy_map_save_path = os.path.join(args.save_path, 'clip', 'relevancy_maps')
    clip_mask_map_save_path = os.path.join(args.save_path, 'clip', 'mask_maps')
    clip_mask_save_path = os.path.join(args.save_path, 'clip', 'masks')
    # clip_mask_object_save_path = os.path.join(args.save_path, 'clip', 'object')
    
    # anything_mask_save_path = os.path.join(args.save_path, 'anything', 'mask_maps')
    # rgb_save_path = os.path.join(args.save_path, 'rgb')
    rendered_feature_save_path = os.path.join(args.save_path, 'rendered_feature')
    instance_feature_save_path = os.path.join(args.save_path, 'instance_feature')
    clip_feature_save_path = os.path.join(args.save_path, 'clip_feature')
    # os.makedirs(point_mask_map_save_path, exist_ok=True)
    # os.makedirs(point_mask_save_path, exist_ok=True)
    # os.makedirs(point_mask_object_save_path, exist_ok=True)
    os.makedirs(clip_relevancy_map_save_path, exist_ok=True)
    os.makedirs(clip_mask_map_save_path, exist_ok=True)
    os.makedirs(clip_mask_save_path, exist_ok=True)
    # os.makedirs(clip_mask_object_save_path, exist_ok=True)
    # os.makedirs(anything_mask_save_path, exist_ok=True)
    os.makedirs(rgb_save_path, exist_ok=True)
    os.makedirs(rendered_feature_save_path, exist_ok=True)
    os.makedirs(instance_feature_save_path, exist_ok=True)
    os.makedirs(clip_feature_save_path, exist_ok=True)
    for i in tqdm(range(len(colmap_eval_cameras))):
        cam = colmap_eval_cameras[i]
        with torch.no_grad():
            render_pkg = render(cam, gaussian, pipe, background)
            image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
            h, w = image_tensor.shape[:2]
            image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            rendered_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
            rendered_clip_feature = gaussian.clip_feature_decoder(rendered_feature[None])[0]
            instance_feature = F.normalize(rendered_feature, dim=0)
            rendered_clip_feature = F.normalize(rendered_clip_feature, dim=0)
            if rendered_feature_pca_dict is None:
                rendered_feature_pca_dict = get_pca_dict(rendered_feature)
            if instance_feature_pca_dict is None:
                instance_feature_pca_dict = get_pca_dict(instance_feature)
            if clip_feature_pca_dict is None:
                clip_feature_pca_dict = get_pca_dict(rendered_clip_feature)


            clip_relevancy_map = clip.get_relevancy(rendered_clip_feature.reshape(-1, h * w).permute(1, 0), 0)[..., 0].reshape(h, w, 1)
            # print(clip_relevancy_map.shape)
            clip_relevancy_map_rgb_tensor = apply_colormap(clip_relevancy_map, ColormapOptions(normalize=True, colormap_min=-1))
            mask = clip_relevancy_map[..., 0] > 0.5
            mask_map = image_tensor.clone()
            mask_map[mask] = mask_map[mask] * 0.5 + torch.tensor([[1, 0, 0]], device='cuda') * 0.5
            mask_map[~mask] /= 2

            image.save(os.path.join(rgb_save_path, f'rgb_{i:03d}.jpg'))
            Image.fromarray((apply_colormap(rendered_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(rendered_feature_save_path, f'{cam.image_name}.jpg'))
            Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(instance_feature_save_path, f'{cam.image_name}.jpg'))
            Image.fromarray((apply_colormap(rendered_clip_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=clip_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(clip_feature_save_path, f'{cam.image_name}.jpg'))
            Image.fromarray((clip_relevancy_map_rgb_tensor.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(clip_relevancy_map_save_path, f'{cam.image_name}.jpg'))
            Image.fromarray((mask_map.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(clip_mask_map_save_path, f'{cam.image_name}.jpg'))
            Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(clip_mask_save_path, f'{cam.image_name}.jpg'))
    # device = "cuda:0"
    # self.colors = np.random.random((500, 3))