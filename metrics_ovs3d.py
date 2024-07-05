import os
import json
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from utils.general_utils import AttrDict

def get_iou(pred_mask, gt_mask):
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum() / union.sum()
    return iou

# def get_iou(pred_mask, gt_mask):
#     pred_mask = pred_mask == 255
#     gt_mask = gt_mask == 255
#     intersection = pred_mask & gt_mask
#     union = pred_mask | gt_mask
#     iou = intersection.sum() / union.sum()
#     return iou

# def get_iou(pred_mask, gt_mask):
#     ious = []
#     intersection_pos = (pred_mask==255) & (gt_mask==255)
#     union_pos = (pred_mask==255) | (gt_mask==255)
#     ious.append(intersection_pos.sum() / union_pos.sum())
#     intersection_neg = (pred_mask==0) & (gt_mask==0)
#     union_neg = (pred_mask==0) | (gt_mask==0)
#     ious.append(intersection_neg.sum() / union_neg.sum())
#     return sum(ious) / len(ious)

def get_accuracy(pred_mask, gt_mask):
    h, w = pred_mask.shape
    # print(((pred_mask==255) & (gt_mask==255)).shape)
    # print((pred_mask==255).sum())
    pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
    # print(pos_acc)
    neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
    # tp_fn = (pred_mask == gt_mask).sum()
    return (pos_acc + neg_acc) / 2

def get_correct(pred_mask, gt_mask):
    return ((pred_mask==255) & (gt_mask==255)).sum()

# def get_accuracy(pred_mask, gt_mask):
#     h, w = pred_mask.shape
#     # print(((pred_mask==255) & (gt_mask==255)).shape)
#     # print((pred_mask==255).sum())
#     pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
#     # print(pos_acc)
#     # neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
#     # tp_fn = (pred_mask == gt_mask).sum()
#     return pos_acc

# def get_accuracy(pred_mask, gt_mask):
#     h, w = pred_mask.shape
#     # print(((pred_mask==255) & (gt_mask==255)).shape)
#     # print((pred_mask==255).sum())
#     pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
#     # print(pos_acc)
#     # neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
#     # tp_fn = (pred_mask == gt_mask).sum()
#     return pos_acc

# def get_pr(pred_mask, gt_mask):
#     # tp = (pred_mask == gt_mask).sum()
#     tp = ((pred_mask & gt_mask) == 255).sum()
#     tp_fp = (pred_mask.reshape(-1) == 255).sum()
#     return tp / tp_fp

def read_segmentation_maps(root_dir, downsample=8):
        segmentation_path = os.path.join(root_dir, 'segmentations')
        classes_file_path = os.path.join(root_dir, 'segmentations', 'classes.txt')
        with open(classes_file_path, 'r') as f:
            classes = f.readlines()
        classes = [class_.strip() for class_ in classes]
        # get a list of all the folders in the directory
        folders = [f for f in os.listdir(segmentation_path) if os.path.isdir(os.path.join(segmentation_path, f))]

        seg_maps = []
        idxes = [] # the idx of the test imgs
        for folder in folders:
            idxes.append(int(folder))  # to get the camera id
            seg_for_one_image = []
            for class_name in classes:
                # check if the seg map exists
                seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
                if not os.path.exists(seg_path):
                    raise Exception(f'Image {class_name}.png does not exist')
                img = Image.open(seg_path).convert('L')
                # resize the seg map
                if downsample != 1.0:
                    img_wh = (int(img.size(0) / downsample), int(img.size(1) / downsample))
                    img = img.resize(img_wh, Image.NEAREST) # [W, H]
                img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
                img = img.flatten() # [H*W]
                seg_for_one_image.append(img)

            seg_for_one_image = np.stack(seg_for_one_image, axis=0)
            seg_for_one_image = seg_for_one_image.transpose(1, 0)
            seg_maps.append(seg_for_one_image)

        seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
        return seg_maps

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='scripts/16_ovs3d_test_config.json')
    parser.add_argument("--pred_path", type=str, default='output/16_ovs3d_masks')
    parser.add_argument("--gt_path", type=str, default='data/ovs3d')
    
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
    scenes = list(cfg.keys())
    # masks = []
    # gt_masks = []
    ious = []
    accs = []
    # prs = []
    metrics = 'Scene\tIoU\tAcc\n'
    for scene in scenes:
        print(scene)
        # scene_cfg = cfg[scene]
        # print(scenes)
        # for prompt in prompts:
        iou = []
        acc = []
        mask_path = os.path.join(args.gt_path, scene, 'segmentations')
        views = os.listdir(mask_path)
        views = [view for view in views if 'txt' not in view]
        # scene_ious = {}
        # scene_accs = {}
        scene_ious = []
        scene_accs = []
        for view in tqdm(views):
            view_masks = os.listdir(os.path.join(mask_path, view))
            view_ious = []
            # view_accs = []
            view_correct = 0
            view_gt = 0
            # view_gt_mask = []
            # view_mask = []
            for view_mask in view_masks:
                gt_mask_path = os.path.join(mask_path, view, view_mask)
                pred_mask_path = os.path.join(args.pred_path, scene, view, 'masks', view_mask)
                mask = np.array(Image.open(pred_mask_path))
                # pixel_cnt = mask.shape[0] * mask.shape[1]
                gt_mask = np.array(Image.open(gt_mask_path).resize((mask.shape[1], mask.shape[0])))[..., 0]
                # mask = (np.array(Image.open(pred_mask_path)) / 255).astype(np.uint8)
                # gt_mask = (np.array(Image.open(gt_mask_path).resize((mask.shape[1], mask.shape[0])))[..., 0] / 255).astype(np.uint8)
                # view_gt_mask.append(gt_mask)
                # view_mask.append(mask)
                
                iou = get_iou(mask, gt_mask)
                # acc = get_accuracy(mask, gt_mask)
                view_ious.append(iou)
                view_correct += (get_correct(mask, gt_mask))
                view_gt += (gt_mask == 255).sum()
                # view_accs.append(acc)
            scene_ious.append(sum(view_ious) / len(view_ious))
            # scene_accs.append(sum(view_accs) / len(view_accs))
            print(view_correct / view_gt)
            scene_accs.append(view_correct / view_gt)
            
                # if scene_ious.get(view_mask, 0) == 0:
                #     scene_ious[view_mask] = [iou]
                #     scene_accs[view_mask] = [acc]
                # else:
                #     scene_ious[view_mask].append(iou)
                #     scene_accs[view_mask].append(acc)
        # scene_iou = []
        # scene_acc = []
        # for key in scene_ious.keys():
        #     scene_iou.append(sum(scene_ious[key]) / len(scene_ious[key]))
        #     scene_acc.append(sum(scene_accs[key]) / len(scene_accs[key]))

        scene_iou = sum(scene_ious) / len(scene_ious)
        scene_acc = sum(scene_accs) / len(scene_accs)
        ious.append(scene_iou)
        accs.append(scene_acc)
        metrics += f'{scene}\t{scene_iou}\t{scene_acc}\n'
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    metrics += f'mean\t{mean_iou}\t{mean_acc}'
    with open(f'ovs3d.txt', 'w') as f:
        f.write(metrics)


