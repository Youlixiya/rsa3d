import os
from PIL import Image
import numpy as np
from argparse import ArgumentParser
def get_iou(pred_mask, gt_mask):
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum() / union.sum()
    return iou

def get_accuracy(pred_mask, gt_mask):
    h, w = pred_mask.shape
    return (pred_mask==gt_mask).sum() / (h * w)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pred_output_path", type=str, default='output/16_llff_masks/qwen_sam')
    parser.add_argument("--gt_path", type=str, default='llff_reasoning_masks')
    parser.add_argument("--tag", type=str, default='qwen_sam_metrics')
    args = parser.parse_args()
    scenes = os.listdir(args.pred_output_path)
    scenes.sort()
    # print(scenes)
    masks = []
    gt_masks = []
    ious = []
    accs = []
    metrics = 'Scene\tIoU\tAcc\tPR\n'
    for scene in scenes:
        prompts = os.listdir(os.path.join(args.pred_output_path, scene))
        prompt = prompts[0]
        # print(scenes)
        # for prompt in prompts:
        iou = []
        acc = []
        mask_path = os.path.join(args.pred_output_path, scene, prompt, 'masks')
        gt_mask_path = os.path.join(args.gt_path, scene, prompt)
        mask_names = os.listdir(mask_path)
        for mask_name in mask_names:
            # mask = (np.array(Image.open(os.path.join(mask_path, mask_name))) * 255).astype(np.uint8)
            mask = np.array(Image.open(os.path.join(mask_path, mask_name)))
            gt_mask = np.array(Image.open(os.path.join(gt_mask_path, mask_name)))
            iou.append(get_iou(mask, gt_mask))
            acc.append(get_accuracy(mask, gt_mask))
        scene_iou = sum(iou) / len(iou)
        scene_acc = sum(acc) / len(acc)
        ious.append(scene_iou)
        accs.append(scene_acc)
        metrics += f'{scene}\t{scene_iou}\t{scene_acc}\n'
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    metrics += f'mean\t{mean_iou}\t{mean_acc}'
    with open(f'{args.tag}.txt', 'w') as f:
        f.write(metrics)


