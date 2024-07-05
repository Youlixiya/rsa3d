import os
import cv2
import torch
import json
import time
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from dashscope import MultiModalConversation
from http import HTTPStatus
from PIL import Image
from PIL import ImageDraw
from copy import deepcopy
import dashscope
import re
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from utils.general_utils import AttrDict
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from lisa.lisa_pipeline import LISAPipeline

COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')

def get_instance_embeddings(gaussian, box, instance_feature):
    feature_dim = instance_feature.shape[0]
    mask_instance_feature = instance_feature[:, box[1]:box[3], box[0]:box[2]].reshape(feature_dim, -1).permute(1, 0)
    instance_embeddings_index = torch.argmax(mask_instance_feature @ gaussian.instance_embeddings.T, dim=-1)
    unique_index, counts = torch.unique(instance_embeddings_index, return_counts=True)
    instance_embedding = gaussian.instance_embeddings[unique_index[torch.argmax(counts)]]
    return instance_embedding

def instance_segmentation(image, gaussian, instance_embeddings, render_instance_feature, mask_threshold, device='cuda'):
    h, w = render_instance_feature.shape[1:]
    t1 = time.time()
    similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    t2 = time.time()
    print(f'time:{t2 - t1}')
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = image.clone()
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + COLORS[i] /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map
def qwen_template(prompt):
    return f'Please grounding <ref> {prompt} </ref>'
def extract_box(text, w, h):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, text)
    box = []
    for match in matches:
        box += match.split(',')
    for i in range(len(box)):
        box[i] = eval(box[i])
    box[0] = int(box[0] / 1000 * w)
    box[1] = int(box[1] / 1000 * h)
    box[2] = int(box[2] / 1000 * w)
    box[3] = int(box[3] / 1000 * h)
    return box
def reasoning_grouding_by_qwen(file_path, text_prompt):
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """

    messages = [{
        'role': 'system',
        'content': [{
            'text': '''
                    You are an AI assistant who is good at making accurate vision grounding based on questions asked
                    '''
        }]
    }, {
        'role':
        'user',
        'content': [
            {
                'image': f'file://{file_path}'
            },
            {
                'text': text_prompt
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-chat-v1', messages=messages)
    image = Image.open(file_path)
    
    # print(response.output.choices)
    # answer = response.output.choices[0].message.content[0]['box']
    answer = response.output.choices[0].message.content
    # result_image = response.output.choices[0].message.content[1]['result_image']
    # result_image = Image.open(requests.get(result_image, stream=True).raw)
    box = extract_box(answer, *(image.size))
    return answer, box

def draw_rectangle(image, rectangle_coordinates, outline_color="red", thickness=2):
    # 打开图像
    image = deepcopy(image)
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    for rectangle_coordinate in rectangle_coordinates:
        draw.rectangle(rectangle_coordinate, outline=outline_color, width=thickness)

    return image

# def point_instance_segmentation(image, gaussian, points, render_instance_feature, mask_threshold, device='cuda'):
#     h, w = render_instance_feature.shape[1:]
#     points = torch.tensor(points, dtype=torch.int64, device=device)
#     instance_embeddings = []
#     t1 = time.time()
#     for point in points:
#         instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
#         instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
#         instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
#         instance_embeddings.append(instance_embedding)
#     instance_embeddings = torch.stack(instance_embeddings)
#     similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
#     masks = (similarity_map > mask_threshold)
#     t2 = time.time()
#     print(f'time:{t2 - t1}')
#     masks_all_instance = masks.any(-1)
#     instance_mask_map = image.clone()
#     instance_object_map = image.clone()
#     for i, mask in enumerate(masks.permute(2, 0, 1)):
#         instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + COLORS[i] /255 * 0.5
#     instance_mask_map[~masks_all_instance, :] /= 2
#     instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
#     return masks_all_instance, instance_mask_map, instance_object_map


if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--lisa", action='store_true')
    parser.add_argument("--lisa_model_type", type=str, default="xinlai/LISA-13B-llama2-v1-explanatory")
    parser.add_argument("--lisa_conv_type", type=str, default="llava_llama_2")
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
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        scene = CamScene(args.colmap_dir, h=h, w=w, eval=True)
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.eval_cameras
        img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    # for i, img in enumerate(imgs_name):
    #     if args.image_name == img:
    #         break
    # cam = colmap_cameras.pop(i)
    if args.lisa:
        lisa_pipeline = LISAPipeline(args.lisa_model_type, local_rank=0, load_in_4bit=False, load_in_8bit=True, conv_type=args.lisa_conv_type)
        save_path = os.path.join(args.save_path, 'lisa', args.reasoning_prompt)
    else:
        save_path = os.path.join(args.save_path, args.reasoning_prompt)
    os.makedirs(save_path, exist_ok=True)
    instance_embeddings = None
    rendered_feature_pca_dict = None
    instance_feature_pca_dict = None
    for i in tqdm(range(len(colmap_cameras))):
        cam = colmap_cameras[i]
        with torch.no_grad():
            rendered_feature_save_path = os.path.join(save_path, 'rendered_features')
            instance_feature_save_path = os.path.join(save_path, 'instance_features')
            mask_map_save_path = os.path.join(save_path, 'mask_maps')
            mask_save_path = os.path.join(save_path, 'masks')
            mask_object_save_path = os.path.join(save_path, 'objects')
            os.makedirs(rendered_feature_save_path, exist_ok=True)
            os.makedirs(instance_feature_save_path, exist_ok=True)
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
            if args.lisa:
                result_list, mask_result_list, mask_list, mask_rgb_list, output_str = lisa_pipeline(lisa_text_prompt, image=image)
                masks_all_instance = mask_result_list[0]
                instance_mask_map = result_list[0] / 255
                instance_object_map = mask_list[0] / 255
            else:
                if instance_embeddings is None:
                    reasoning_prompt = qwen_template(args.reasoning_prompt)
                    image.save('tmp.jpg')
                    tmp_image_path = os.path.join(os.getcwd(), 'tmp.jpg')
                    answer, box = reasoning_grouding_by_qwen(tmp_image_path, reasoning_prompt)
                    reasoning_result = draw_rectangle(image, [box])
                    reasoning_result.save(os.path.join(args.save_path, 'reasoning_result.jpg'))
                    instance_embeddings = get_instance_embeddings(gaussian, box, instance_feature)
                masks_all_instance, instance_mask_map, instance_object_map = instance_segmentation(image_tensor, gaussian, instance_embeddings, instance_feature, args.mask_threshold, device='cuda')
            Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(rendered_feature_save_path, f'{cam.image_name}.png'))
            Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(instance_feature_save_path, f'{cam.image_name}.png'))
            Image.fromarray((masks_all_instance.cpu().numpy() * 255).astype(np.uint8), mode='L').save(os.path.join(mask_save_path, f'{cam.image_name}.png'))
            Image.fromarray((instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(mask_map_save_path, f'{cam.image_name}.png'))
            Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(mask_object_save_path, f'{cam.image_name}.png'))