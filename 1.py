import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, LlamaTokenizer

model_path = "ckpts/CogVLM-grounding-generalist-hf-quant4"


tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# chat example
# expr = "what is made of stone"
expr = "camera"
query = f'Can you point out {expr} in the image and provide the bounding boxes of its location?'
image = Image.open("data/ovs3d/bed/images_4/00.jpg").convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    responese = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(responese)
box = eval(responese)[0]
height, width = image.height, image.width
if height > width:
    box[0] = box[0] / 1000 * width
    box[1] = box[1] / 1000 * height
    box[2] = box[2] / 1000 * width
    box[3] = box[3] / 1000 * height
from PIL import ImageDraw
img_box = image.copy()
draw = ImageDraw.Draw(img_box)
draw.rectangle(box, outline='red', width=2)
box = np.array(box)

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




# example output 
# a room with a ladder [[378,107,636,998]] and a blue and white towel [[073,000,346,905]].</s>
# NOTE: The model's squares have dimensions of 1000 by 1000, which is important to consider.
