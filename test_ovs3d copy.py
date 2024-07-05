import os
import cv2
import torch
import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils.general_utils import AttrDict
from PIL import Image
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from scene.clip_encoder import OpenCLIPNetwork, OpenCLIPNetworkConfig
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')

@torch.no_grad()
def select_semantic_embeddings(clip_model, gaussian, text_prompt, neg_features, text_query_threshold, device='cuda'):
    
    text_prompt = clip.tokenize(text_prompt.split(',')).to(device)
    # text_prompt = alpha_clip.tokenize([self.text_prompt.value]).to(self.device)
    pos_features = clip_model.encode_text(text_prompt)
    pos_features /= pos_features.norm(dim=-1, keepdim=True)
    
    total_features = torch.cat([neg_features, pos_features])
    total_mm = gaussian.clip_embeddings @ total_features.T
    pos_mm = total_mm[:, 1:]
    neg_mm = total_mm[:, [0]].repeat(1, pos_mm.shape[-1])
    # print(pos_mm.shape)
    # print(pos_mm.shape)
    total_similarity = torch.stack([pos_mm, neg_mm], dim=-1)
    softmax = (100 * total_similarity).softmax(-1)
    pos_softmax = softmax[..., 0]
    valid_mask = pos_softmax > text_query_threshold
    semantic_valid_num = valid_mask.sum(0)
    semantic_embeddings = []
    for i in range(valid_mask.shape[-1]):
        semantic_embeddings.append(gaussian.instance_embeddings[valid_mask[:, i], :])
    semantic_embeddings = torch.cat(semantic_embeddings)
    return semantic_valid_num, semantic_embeddings

def text_semantic_segmentation(image, semantic_embeddings, instance_feature, mask_threshold, semantic_valid_num=None, device='cuda'):
    similarity_map = (instance_feature.reshape(-1, h * w).permute(1, 0) @ semantic_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all = masks.any(-1)
    semantic_mask_map = image.clone()
    # sematic_object_map = image.clone()
    sematic_object_map = torch.cat([image.clone(), masks_all[..., None]], dim=-1)
    start_index = 0
    # print(semantic_valid_num)
    # print(semantic_embeddings.shape)
    for i in range(len(semantic_valid_num)):
        mask = masks[..., start_index:start_index + semantic_valid_num[i]].any(-1)
        # print(mask.shape)
        semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + COLORS[i] / 255 * 0.5
        # semantic_mask_map[~mask, :] = semantic_mask_map[~mask, :] * 0.5 + torch.tensor([0, 0, 0], device=self.device) * 0.5
        start_index += semantic_valid_num[i]
    semantic_mask_map[~masks_all, :] /= 2
    # sematic_object_map[~masks_all, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    # sematic_object_map[masks_all, -1] = torch.tensor([1], dtype=torch.float32, device=device)
    
    return masks_all, semantic_mask_map, sematic_object_map

def get_instance_embeddings(gaussian, points, render_instance_feature, device='cuda'):
    h, w = render_instance_feature.shape[1:]
    points = torch.tensor(points, dtype=torch.int64, device=device)
    instance_embeddings = []
    for point in points:
        instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
        instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
        instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
        instance_embeddings.append(instance_embedding)
    instance_embeddings = torch.stack(instance_embeddings)
    return instance_embeddings
    
def point_instance_segmentation(image, gaussian, instance_embeddings, render_instance_feature, mask_threshold, device='cuda'):
    similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = image.clone()
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + torch.tensor(COLORS[i], dtype=torch.float32, device=device) /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map

def instance_segmentation_all(gaussian, render_instance_feature):
    h, w = render_instance_feature.shape[1:]
    instance_index = torch.argmax((F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
    # print(instance_index)
    instance_masks = gaussian.instance_colors[instance_index].reshape(h, w, 3)
    return instance_masks

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = AttrDict(json.load(f)[args.scene])
    args = AttrDict(args.__dict__)
    args.update(cfg)
    gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
    gaussian.load_ply(args.gs_source)
    if args.feature_gs_source:
        gaussian.load_feature_params(args.feature_gs_source)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
    colmap_cameras = None
    render_cameras = None
    target_image_names = args.image_names
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        # if args.h == -1 and args.w == -1:
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        # else:
        #     h = args.h
        #     w = args.w
        scene = CamScene(args.colmap_dir, h=h, w=w)
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras
        img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    instance_embeddings = None
    rendered_feature_pca_dict = None
    instance_feature_pca_dict = None
    #clip
    clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    clip.set_positives(args.clip_text_prompt)
    for i in tqdm(range(len(colmap_cameras))):
        cam = colmap_cameras[i]
        if cam.image_name not in target_image_names:
            continue
        with torch.no_grad():
            mask_map_save_path = os.path.join(args.save_path, cam.image_name, 'mask_maps')
            mask_save_path = os.path.join(args.save_path, cam.image_name, 'masks')
            mask_object_save_path = os.path.join(args.save_path, cam.image_name, 'objects')
            os.makedirs(mask_map_save_path, exist_ok=True)
            os.makedirs(mask_save_path, exist_ok=True)
            os.makedirs(mask_object_save_path, exist_ok=True)
            render_pkg = render(cam, gaussian, pipe, background)
            image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
            image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
            instance_feature = F.normalize(render_feature.reshape(-1, h*w), dim=0).reshape(-1, h, w)
            if rendered_feature_pca_dict is None:
                rendered_feature_pca_dict = get_pca_dict(render_feature)
            if instance_feature_pca_dict is None:
                instance_feature_pca_dict = get_pca_dict(instance_feature)
            Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'rendered_feature_pca.jpg'))
            Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'instance_feature_pca.jpg'))
            for j in range(len(clip.positives)):
                relevancy = clip.get_relevancy(gaussian.clip_embeddings, j)[..., 0]
                print(relevancy.max())
                mask_index = relevancy > args.text_query_threshold[j]
                instance_embeddings = gaussian.instance_embeddings[mask_index]
                semantic_valid_num = [len(instance_embeddings)]
                masks_all_semantic, semantic_mask_map, semantic_object_map = text_semantic_segmentation(image_tensor, instance_embeddings, instance_feature, args.text_mask_threshold[j], semantic_valid_num)
                clip_mask = (masks_all_semantic.cpu().numpy() * 255).astype(np.uint8)
                clip_mask_map = (semantic_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                clip_object = (semantic_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                text_prompt = args.clip_text_prompt[j]
                Image.fromarray(clip_mask).save(os.path.join(mask_save_path, f'{text_prompt}.png'))
                Image.fromarray(clip_mask_map).save(os.path.join(mask_map_save_path, f'{text_prompt}.png'))
                Image.fromarray(clip_object).save(os.path.join(mask_object_save_path, f'{text_prompt}.png'))
                # clip_semantic_embeddings = instance_embeddings[mask_index]
            # if instance_embeddings == None:
            #     instance_embeddings = get_instance_embeddings(gaussian, args.points, render_feature)
            # total_rendered_feature = [render_feature]
            # if gaussian.rgb_decode:
            #     total_rendered_feature.append(render_pkg['render'])
            # if gaussian.depth_decode:
            #     total_rendered_feature.append(render_pkg['depth_3dgs'])
            # total_rendered_feature = torch.cat(total_rendered_feature, dim=0)
            # h, w = total_rendered_feature.shape[1:]
            # total_rendered_feature = total_rendered_feature.reshape(-1, h*w).permute(1, 0)
            # if gaussian.feature_aggregator:
            #     total_rendered_feature = F.normalize(gaussian.feature_aggregator(total_rendered_feature), dim=-1)
            # else:
            #     total_rendered_feature = F.normalize(total_rendered_feature, dim=-1)
            # total_rendered_feature = total_rendered_feature.permute(1, 0).reshape(-1, h, w)
            # masks_all_instance, instance_mask_map, instance_object_map = point_instance_segmentation(image_tensor, gaussian, instance_embeddings, render_feature, args.mask_threshold, device='cuda')
            # instance_masks = instance_segmentation_all(gaussian, render_feature)
            # image.save(os.path.join(args.save_path, f'rendered_rgb_{args.image_name}'))
            # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'rendered_feature_pca_{args.image_name}'))
            # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'total_rendered_feature_pca_{args.image_name}'))
            # Image.fromarray(np.stack([(masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(mask_save_path, f'mask_{i:03d}.png'))
            # Image.fromarray((instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(mask_map_save_path, f'mask_map_{i:03d}.jpg'))
            # Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_object_map_{args.image_name}'))
            # Image.fromarray((instance_masks.cpu().numpy()).astype(np.uint8)).save(os.path.join(mask_save_path, f'mask_{i}.png'))
    # device = "cuda:0"
    # self.colors = np.random.random((500, 3))