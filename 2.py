from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import numpy as np

processor = LlavaNextProcessor.from_pretrained("ckpts/llava-v1.6-vicuna-13b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("ckpts/llava-v1.6-vicuna-13b-hf",
                                                          torch_dtype=torch.float16,
                                                          low_cpu_mem_usage=True,
                                                          load_in_4bit=True,
                                                          use_flash_attention_2=True) 
# model.to("cuda:0")
print(model.device)
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
url = "data/ovs3d/sofa/images_4/00.jpg"
image = Image.open(url)
#bed
# text = "Please provide the bounding box coordinate of the region this sentence describes: what can be used to take photos."
# text = "Please provide the bounding box coordinate of the region this sentence describes: what is the part of person."
# text = "Please provide the bounding box coordinate of the region this sentence describes: what is the yellow fruit."
# text = "Please provide the bounding box coordinate of the region this sentence describes: what can be worn on the foot."
# text = "Please provide the bounding box coordinate of the region this sentence describes: what can be used to carry wallet."
# text = "Please provide the bounding box coordinate of the region this sentence describes: where to sleep."
#bench
# text = "Please provide the bounding box coordinate of the region this sentence describes: who has blond hair."
text = "Please provide the bounding box coordinate of the region this sentence describes: what is a robot toy."
# text = "Please provide the bounding box coordinate of the region this sentence describes: what is made of stone."

prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{text} ASSISTANT:"


inputs = processor(prompt, image, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(output[0], skip_special_tokens=True)
response = response.split(' ASSISTANT:')[-1]
print(response)

from PIL import ImageDraw
max_edge = max((image.width, image.height))
img_box = image.copy()
draw = ImageDraw.Draw(img_box)
if image.width < image.height:
    x_origin = (image.height - image.width) // 2
    y_origin = 0
else:
    x_origin = 0
    y_origin = (image.width - image.height) // 2
x1, y1, x2, y2 = eval(response)
x1 = x1 * max_edge - x_origin
y1 = y1 * max_edge - y_origin
x2 = x2 * max_edge - x_origin
y2 = y2 * max_edge - y_origin
box = np.array([x1, y1, x2, y2])
draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
img_box.save("box_tmp.jpg")
predictor.set_image(np.array(image))
masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=box[None, :],
    multimask_output=False,
)
mask = masks[0]
masks_all_instance = (mask * 255).astype(np.uint8)

instance_mask_map = np.array(image)
instance_object_map = np.array(image)
instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + np.array([255, 0, 0]) * 0.5
instance_object_map[~mask, :] = np.array([255, 255, 255])
Image.fromarray(instance_object_map).save('mask_tmp.jpg')
# prepare image and text prompt, using the appropriate prompt template

