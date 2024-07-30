import os
import re
import cv2
import torch
import json
from dashscope import MultiModalConversation
from http import HTTPStatus
import dashscope
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torchvision
import torch.nn.functional as F
from utils.general_utils import AttrDict
import open_clip
from PIL import Image, ImageDraw
from sklearn.metrics import jaccard_score, accuracy_score
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from scene.clip_encoder import OpenCLIPNetwork, OpenCLIPNetworkConfig
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from transformers import AutoModelForCausalLM, AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor
COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')
torch.manual_seed(1234)
# def qwen_template0(prompt):
#     return f'Please output the name of the object that best matches the following text description. The text description : {prompt}.'

qwen_template0 = "Infer the detail name of the target object from the text input, and give an explanation for the inferred result. Format of the output {'category':, 'explanation':}. The text description : "

def qwen_template1(prompt):
    return f'Please grounding <ref> {prompt} </ref>'

def clip_template(prompt):
    return f'a photo of {prompt}'

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

def get_box_by_mask(mask):
    non_zero_indices = torch.nonzero(mask.float())
    min_indices = torch.min(non_zero_indices, dim=0).values
    max_indices = torch.max(non_zero_indices, dim=0).values
    top_left = min_indices
    bottom_right = max_indices + 1
    return top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()

def draw_rectangle(image, rectangle_coordinates, outline_color="red", thickness=2):
    # 打开图像
    image = deepcopy(image)
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    for rectangle_coordinate in rectangle_coordinates:
        draw.rectangle(rectangle_coordinate, outline=outline_color, width=thickness)

    return image

def reasoning_grouding_by_qwen(file_path, text_prompt):
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """

    messages = [{
        'role': 'system',
        'content': [{
            'text': '''
                    You are an AI assistant that is good at locating objects that best match the text description
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
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    image = Image.open(file_path)
    
    # print(response.output.choices)
    # answer = response.output.choices[0].message.content[0]['box']
    answer = response.output.choices[0].message.content[0]['box']
    print(answer)
    # result_image = response.output.choices[0].message.content[1]['result_image']
    # result_image = Image.open(requests.get(result_image, stream=True).raw)
    box = extract_box(answer, *(image.size))
    return answer, box, image

def reasoning_by_qwen(file_path, text_prompt):
    """Sample of use local file.
       linux&mac file schema: file:///home/images/test.png
       windows file schema: file://D:/images/abc.png
    """

    messages = [{
        
        'role': 'system',
        'content': [{
            'text': '''
                   You are an AI assistant that is good at determining whether there is an object in the scene that best matches the description of the text, returning yes if you can observe it, and no otherwise.
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
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    image = Image.open(file_path)
    
    # print(response.output.choices)
    answer = response.output.choices[0].message.content[0]['text']
    print(answer)
    # result_image = response.output.choices[0].message.content[1]['result_image']
    # result_image = Image.open(requests.get(result_image, stream=True).raw)
    return answer

def get_masks_in_box(gaussian, box, image_np, instance_feature):
    feature_dim, h, w = instance_feature.shape
    mask_instance_feature = instance_feature[:, box[1]:box[3], box[0]:box[2]].reshape(feature_dim, -1).permute(1, 0)
    instance_embeddings_index = torch.argmax(mask_instance_feature @ gaussian.instance_embeddings.T, dim=-1)
    instance_index_map = (gaussian.instance_embeddings @ instance_feature.reshape(feature_dim, h * w)).argmax(0)
    instance_index_map = instance_index_map.reshape(h, w)
    print(instance_index_map.shape)
    unique_indexes, counts = torch.unique(instance_embeddings_index, return_counts=True)
    mask_images_np = []
    new_unique_indexes = []
    # return unique_indexes
    for index in unique_indexes:
        mask_bool = (instance_index_map == index)
        # print(mask_bool.shape)
        # mask_box = get_box_by_mask(mask_bool)
        mask_image_np = image_np.copy()
        mask_bool = mask_bool.cpu().numpy()
        mask_image_np[~mask_bool, :] = np.array([0, 0, 0])
        mask_box = mask_bool[box[1]: box[3], box[0]: box[2]]
        mask_area = mask_box.sum()
        box_area = (box[3] - box[1]) * (box[2] - box[0])
        if (mask_area / box_area) < 0.1:
            continue
        mask_images_np.append(mask_image_np[box[1]: box[3], box[0]: box[2], :])
        new_unique_indexes.append(index)
    # return new_unique_indexes
    return new_unique_indexes, mask_images_np

def get_instance_embeddings_by_box(gaussian, box, instance_feature):
    feature_dim = instance_feature.shape[0]
    mask_instance_feature = instance_feature[:, box[1]:box[3], box[0]:box[2]].reshape(feature_dim, -1).permute(1, 0)
    instance_embeddings_index = torch.argmax(mask_instance_feature @ gaussian.instance_embeddings.T, dim=-1)
    unique_index, counts = torch.unique(instance_embeddings_index, return_counts=True)
    return unique_index[torch.argmax(counts)]

def get_instance_embeddings_by_mask(gaussian, mask, instance_feature):
    feature_dim = instance_feature.shape[0]
    mask_instance_feature = instance_feature[:, mask].reshape(feature_dim, -1).permute(1, 0)
    instance_embeddings_index = torch.argmax(mask_instance_feature @ gaussian.instance_embeddings.T, dim=-1)
    unique_index, counts = torch.unique(instance_embeddings_index, return_counts=True)
    return unique_index[torch.argmax(counts)]

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

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    dt = (dt>128).astype('uint8')
    gt = (gt>128).astype('uint8')
    

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return boundary_iou


def load_mask(mask_path):
    """Load the mask from the given path."""
    return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale

def resize_mask(mask, target_shape):
    """Resize the mask to the target shape."""
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def calculate_iou(mask1, mask2):
    """Calculate IoU between two boolean masks."""
    mask1_bool = mask1 > 128
    mask2_bool = mask2 > 128
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    iou = np.sum(intersection) / np.sum(union)
    return iou

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--train_views", type=int, default=1)
    parser.add_argument("--match_type", type=str, default='clip')
    
    # parser.add_argument("--scene", type=str, required=True)
    # parser.add_argument("--gt_path", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--scene", type=str, required=True)
    # scene_names = ['figurines', 'ramen', 'teatime']
    scene_names = ['teatime']
    args = parser.parse_args()
    metrics = 'Scene\tIoU\tAcc\n'

    #sam
    sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda:0"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam.to(device=device)

    predictor = SamPredictor(sam)

    #clip
    preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16",
                                                            pretrained="laion2b_s34b_b88k",
                                                            precision="fp16",
                                                            device='cuda:0')
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    clip_model = clip_model.cuda()
    all_IoUs = []
    all_accuracies = []
    for scene_name in tqdm(scene_names):
        with open(args.cfg_path, 'r') as f:
            cfg = AttrDict(json.load(f)[scene_name])
        args = AttrDict(args.__dict__)
        args.update(cfg)
        gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
        gaussian.load_ply(args.gs_source)
        if args.feature_gs_source:
            gaussian.load_feature_params(args.feature_gs_source)
        # print(gaussian.clip_embeddings)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
        colmap_cameras = None
        render_cameras = None

        if args.colmap_dir is not None:
            img_root = os.path.join(args.colmap_dir, args.images)
            img_name = os.listdir(img_root)[0]
            img_suffix = img_name.split('.')[-1]
            
            # if args.h == -1 and args.w == -1:
            h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
            # else:
            #     h = args.h
            #     w = args.w
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
        
        instance_embeddings = []
        rendered_feature_pca_dict = None
        instance_feature_pca_dict = None
        iou_scores = {}  # Store IoU scores for each class
        biou_scores = {}
        

        reasoning_prompts = args.reasoning_prompts
        clip_text_prompts = args.clip_text_prompts
        # clip2reasoning_dict = {}
        reasoning2clip_dict = {}
        for i in range(len(reasoning_prompts)):
            # clip2reasoning_dict[clip_text_prompts[i]] = reasoning_prompts[i]
            reasoning2clip_dict[reasoning_prompts[i]] = clip_text_prompts[i]
        print(reasoning2clip_dict)
        valid_index = 0
        
        metrics += f'{scene_name}\n'
        for i in tqdm(range(len(reasoning_prompts))):
            reasoning_prompt = reasoning_prompts[i]
            clip_text_prompt = reasoning2clip_dict[reasoning_prompt]
            total_views = 0
            similarity_scores = []
            instance_indexes = []
            for j in tqdm(range(len(colmap_train_cameras))):
                if total_views == args.train_views:
                    break
                cam = colmap_train_cameras[j]
                with torch.no_grad():
                    render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
                    instance_feature = F.normalize(render_feature.reshape(-1, h*w), dim=0).reshape(-1, h, w)
                    if rendered_feature_pca_dict is None:
                        rendered_feature_pca_dict = get_pca_dict(render_feature)
                    if instance_feature_pca_dict is None:
                        instance_feature_pca_dict = get_pca_dict(instance_feature)
                    os.makedirs(os.path.join(args.save_path, cam.image_name), exist_ok=True)
                    Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'rendered_feature_pca.jpg'))
                    Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'instance_feature_pca.jpg'))
                    
                    image_name = cam.image_name
                    image_path = os.path.join(img_root, f'{cam.image_name}.{img_suffix}')
                    save_path = os.path.join(args.save_path, cam.image_name, reasoning_prompt)
                    cache_save_path = os.path.join(save_path, 'cache.pt')

                    cache = {}
                    
                    
                    reasoning_results_save_path = os.path.join(save_path, 'reasoning_result.png')
                    os.makedirs(save_path, exist_ok=True)
                    if args.match_type == 'box': 
                        if os.path.exists(cache_save_path):
                            cache = torch.load(cache_save_path)
                            box = cache['box']
                        else:
                            try:
                                _, box, image_pil = reasoning_grouding_by_qwen(image_path, qwen_template1(reasoning_prompt))
                            except:
                                continue
                        instance_indexes.append(get_instance_embeddings_by_box(gaussian, box, instance_feature))
                        total_views += 1
                        continue
                    if os.path.exists(cache_save_path):
                        cache = torch.load(cache_save_path)
                        seleced_instance_index = cache['seleced_instance_index']

                    else:
                        try:
                            category_explanation = eval(reasoning_by_qwen(image_path, qwen_template0 + reasoning_prompt))
                            category = category_explanation['category']
                            explanation = category_explanation['explanation']
                            _, box, image_pil = reasoning_grouding_by_qwen(image_path, qwen_template1(reasoning_prompt))
                        except:
                            continue
                        reasoning_result = draw_rectangle(image_pil, [box])
                        reasoning_result.save(reasoning_results_save_path)
                        image_np = np.array(image_pil)

                        unique_indexes, mask_images_np = get_masks_in_box(gaussian, box, image_np, instance_feature)
                        mask_clip_embeddings = gaussian.clip_embeddings[unique_indexes, :].half()
                        # unique_indexes = get_masks_in_box(gaussian, box, image_np, instance_feature)
                        # mask_clip_embeddings = []
                        for k in range(len(unique_indexes)):
                            mask_image_np = mask_images_np[k]
                            mask_image_pil = Image.fromarray(mask_image_np)
                            mask_image_pil.save(os.path.join(save_path, f'mask_image_{k}.png'))
                        #     mask_image_tensor = preprocess(mask_image_pil).half().cuda()[None]
                        #     mask_image_clip_embedding = clip_model.encode_image(mask_image_tensor)
                        #     mask_image_clip_embedding_norm = mask_image_clip_embedding / mask_image_clip_embedding.norm(dim=-1, keepdim=True)
                        #     mask_clip_embeddings.append(mask_image_clip_embedding_norm)
                        category_embeddings = clip_model.encode_text(tokenizer([clip_template(category)]).to('cuda:0'))
                        category_embeddings /= category_embeddings.norm(dim=-1, keepdim=True)
                        # mask_clip_embeddings = torch.cat(mask_clip_embeddings)
                        similarity_score = category_embeddings @ mask_clip_embeddings.T
                        similarity_score = torch.softmax(similarity_score, dim=1)
                        print(similarity_score)
                        # similarity_score = similarity_score[[0], :]
                        selected_index = similarity_score.argmax(-1)
                        seleced_instance_index = unique_indexes[selected_index]
                        cache['category'] = category
                        cache['explanation'] = explanation
                        cache['box'] = box
                        cache['category_embeddings'] = category_embeddings
                        cache['mask_clip_embeddings'] = mask_clip_embeddings
                        cache['similarity_score'] = similarity_score
                        cache['seleced_instance_index'] = seleced_instance_index
                        torch.save(cache, cache_save_path)
                    instance_indexes.append(seleced_instance_index)
                    total_views += 1
                        
            # similarity_scores = torch.tensor(similarity_scores)
            instance_indexes = torch.tensor(instance_indexes)
            # print(similarity_scores)
            print(instance_indexes)
            instance_embeddings = []
            unique_index, counts = torch.unique(instance_indexes, return_counts=True)
            index = unique_index[torch.argmax(counts)].long()
            instance_embedding = gaussian.instance_embeddings[index]
            instance_embeddings.append(instance_embedding)
            instance_embeddings = torch.stack(instance_embeddings)
            print(index)

            for j in tqdm(range(len(colmap_eval_cameras))):
                cam = colmap_eval_cameras[j]
                test_index = cam.image_name.split('.')[0].split('_')[-1]
                gt_mask_path = os.path.join(args.colmap_dir, 'test_mask', test_index, f'{clip_text_prompt}.png')
                
                try:
                    gt_mask = load_mask(gt_mask_path)
                except:
                    continue
                with torch.no_grad():
                    mask_map_save_path = os.path.join(args.save_path, cam.image_name, 'mask_maps')
                    mask_save_path = os.path.join(args.save_path, cam.image_name, 'masks')
                    os.makedirs(mask_map_save_path, exist_ok=True)
                    os.makedirs(mask_save_path, exist_ok=True)
                    # os.makedirs(args.save_path, cam.image_name, exist_ok=True)
                    render_pkg = render(cam, gaussian, pipe, background)
                    image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
                    H, W = image_tensor.shape[:2]
                    rgb = image_tensor.reshape(-1, 3)
                    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                    render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
                    instance_feature = F.normalize(render_feature.reshape(-1, h*w), dim=0).reshape(-1, h, w)
                    if rendered_feature_pca_dict is None:
                        rendered_feature_pca_dict = get_pca_dict(render_feature)
                    if instance_feature_pca_dict is None:
                        instance_feature_pca_dict = get_pca_dict(instance_feature)
                    Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'rendered_feature_pca.jpg'))
                    Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, cam.image_name, f'instance_feature_pca.jpg'))
                    
                    # instance_index_map = (gaussian.instance_embeddings @ instance_feature.reshape(-1, h*w)).argmax(0).reshape(H, W)
                    # pred_mask_bool = (index == instance_index_map).cpu().numpy()
                    pred_mask_bool = ((instance_embeddings @ instance_feature.reshape(-1, h*w))[0].reshape(H, W) > args.mask_threshold[i]).cpu().numpy()
                    pred_mask = (pred_mask_bool * 255).astype(np.uint8)
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = resize_mask(pred_mask, gt_mask.shape)
                    iou = calculate_iou(gt_mask, pred_mask)
                    biou = boundary_iou(gt_mask, pred_mask)
                    # print(p_class.shape)
                    if reasoning_prompt not in iou_scores:
                        iou_scores[reasoning_prompt] = []
                        biou_scores[reasoning_prompt] = []
                    iou_scores[reasoning_prompt].append(iou)
                    biou_scores[reasoning_prompt].append(biou)
                    mask_map = image_np.copy().astype(np.float32)
                    mask_map[pred_mask_bool, :] = mask_map[pred_mask_bool, :] * 0.5 + np.array([255, 0, 0]) * 0.5
                    mask_map[~pred_mask_bool, :] /= 2
                    mask_map = mask_map.astype(np.uint8)
                    Image.fromarray(mask_map).save(os.path.join(mask_map_save_path, f'{reasoning_prompt}_{args.match_type}_{args.train_views}.png'))
                    Image.fromarray(pred_mask).save(os.path.join(mask_save_path, f'{reasoning_prompt}_{args.match_type}_{args.train_views}.png'))
            
            print('iou for classes:', np.mean(iou_scores[reasoning_prompt]), 'mean iou:', np.mean(biou_scores[reasoning_prompt]))
                
            metrics += f'{reasoning_prompt}\t{np.mean(iou_scores[reasoning_prompt])}\t{np.mean(biou_scores[reasoning_prompt])}\n'
        
        mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
        mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}
        overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
        overall_mean_biou = np.mean(list(mean_biou_per_class.values()))
        metrics += f'mean\t{overall_mean_iou}\t{overall_mean_biou}\n'
    save_name = f'gsgrouping_{args.match_type}_{args.train_views}.txt'
    with open(save_name, 'w') as f:
        f.write(metrics)
                    # semantic_valid_num = [len(instance_embeddings)]
                    # masks_all_semantic, semantic_mask_map, semantic_object_map = text_semantic_segmentation(image_tensor, instance_embeddings, instance_feature, args.text_mask_threshold[j], semantic_valid_num)
                    # clip_mask = (masks_all_semantic.cpu().numpy() * 255).astype(np.uint8)
                    # clip_mask_map = (semantic_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    # clip_object = (semantic_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    # text_prompt = args.clip_text_prompt[j]
                    # Image.fromarray(clip_mask).save(os.path.join(mask_save_path, f'{text_prompt}.png'))
                    # Image.fromarray(clip_mask_map).save(os.path.join(mask_map_save_path, f'{text_prompt}.png'))
                    # Image.fromarray(clip_object).save(os.path.join(mask_object_save_path, f'{text_prompt}.png'))
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
    