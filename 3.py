from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torchvision
import open_clip
import re
from PIL import Image
import numpy as np
torch.manual_seed(1234)

def get_box_by_mask(mask):
    non_zero_indices = torch.nonzero(mask.float())
    min_indices = torch.min(non_zero_indices, dim=0).values
    max_indices = torch.max(non_zero_indices, dim=0).values
    top_left = min_indices
    bottom_right = max_indices + 1
    return top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()

def qwen_template0(prompt):
    return f'Please determine whether there is an object corresponding to the following text description in the scene. If there is, output the string "yes", otherwise output the string "no". {prompt}.'

def qwen_template1(prompt):
    return f'Output the bounding box of the object that best matches the text description, not the bounding box of an irrelevant object. Please grounding <ref> {prompt} </ref>'

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

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("ckpts/Qwen-VL-Chat", trust_remote_code=True)

# use cuda device
model = AutoModelForCausalLM.from_pretrained("ckpts/Qwen-VL-Chat",
                                             device_map="cuda:4",
                                             trust_remote_code=True,
                                             bf16=True).eval()

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda:3"

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
                                                        device='cuda:3')

# 1st dialogue turn
prompt = "which is red fruit"
image_path = 'data/gsgrouping/figurines/images/frame_00003.jpg'
query = tokenizer.from_list_format([
    {'image': image_path},
    {'text': qwen_template0(prompt)},
])

# inputs = tokenizer(query, return_tensors='pt')
# inputs = inputs.to(model.device)
# pred = model.generate(**inputs)
# response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

# 2nd dialogue turn
# response, history = model.chat(tokenizer, '输出"击掌"的检测框', history=history)
# print(response)

# print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
  image.save('result_qwen.jpg')
else:
  print("no box")


image_pil = Image.open(image_path)
image_np = np.array(image_pil)
box = extract_box(response, image_pil.width, image_pil.height)

predictor.set_image(image_np)
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=np.array(box)[None, :],
    multimask_output=False,
)
mask = torch.from_numpy(masks[0])
bounding_box_image_np = image_np[box[1]: box[3], box[0]: box[2], :]
image_mask_np = image_np.copy()
image_mask_np[~mask, :] = np.array([0, 0, 0])
mask_bounding_box = get_box_by_mask(mask)
bounding_box_image_mask_np = image_mask_np[mask_bounding_box[1]:mask_bounding_box[3], mask_bounding_box[0]:mask_bounding_box[2], :]
bouning_box_image_pil = Image.fromarray(bounding_box_image_np)
bounding_box_image_mask_pil = Image.fromarray(bounding_box_image_mask_np)
bouning_box_image_pil.save(f'result_reasoning_box.png')
bounding_box_image_mask_pil.save(f'result_reasoning_mask.png')
bounding_box_image_mask_tensor = preprocess(bounding_box_image_mask_pil).half().to(device)[None]
bounding_box_image_mask_clip_embedding = clip_model.encode_image(bounding_box_image_mask_tensor)
bounding_box_image_mask_clip_embedding_norm = bounding_box_image_mask_clip_embedding / bounding_box_image_mask_clip_embedding.norm(dim=-1, keepdim=True)