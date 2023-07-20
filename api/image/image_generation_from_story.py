from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import os
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import re
import pickle
import glob
from pyiqa import create_metric

def brisk(image):

    metric_name = "brisque"

    iqa_model = create_metric(metric_name, metric_mode="NR")

    score = iqa_model(image)
    
    return score.item()


story_folder_path = "Default"
for file_path in glob.glob(r"stories/*.pkl"):
    story_folder_path = re.split(r'[\/\.]', file_path)[1]
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

print("Story folder: ",story_folder_path)
# print(data.keys())
# print(data["story"])

story_prompts = []
story_title = data["cover_image_prompt"]
story_prompts.append(story_title)

story_prompts_polarity= ["neutral"]
for story_paragraph in data["story"]:
    story_prompts.append(story_paragraph["image_prompt"])
    story_prompts_polarity.append(story_paragraph["polarity"])
 
    
print(f"Story Title: {story_title}")

print(story_prompts)


DEVICE_CPU = torch.device('cpu:0')
DEVICE_GPU_0 = torch.device('cuda:0')
DEVICE_GPU_1 = torch.device('cuda:1')

print("*** Loading encoder ***")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    subfolder='image_encoder',
    cache_dir='./kand22',
    local_files_only = True
).to(DEVICE_CPU)

print("*** Loading unet ***")
unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    subfolder='unet',
    cache_dir='./kand22',
    local_files_only = True
).half().to(DEVICE_GPU_0)

print("*** Loading prior ***")
prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder, 
    torch_dtype=torch.float32,
    cache_dir='./kand22',
    local_files_only = True
).to(DEVICE_CPU)

print("*** Loading decoder ***")
decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    unet=unet,
    torch_dtype=torch.float16,
    cache_dir='./kand22',
    local_files_only = True
).to(DEVICE_GPU_0)


num_batches = 1
images_per_batch = 1
total_num_images = images_per_batch * num_batches

negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, \
    duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
        mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured"

current_prompt_index = 0
prompt_text = f"{story_prompts[current_prompt_index]} Child Storybook illustration, 4k"

images = []

os.makedirs(story_folder_path, exist_ok = True)

print(f"*** Generating {total_num_images} image(s) ***")
for i in range(len(story_prompts)):
    print(f"* Batch {i} of {len(story_prompts)- 1} *")

    print(f"Prompt: \n{prompt_text}" )

    num_inference_steps = 50 
    
    # Generating embeddings on the CPU
    img_emb = prior(
        prompt=prompt_text,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=images_per_batch)

    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=images_per_batch
    )

    # Converting fp32 to fp16, to run decoder on the GPU
    image_batch = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=num_inference_steps, height=512, width=512)
    
    for i,image in enumerate(image_batch.images):
        image_batch.images[i].save(f"{story_folder_path}/img_{current_prompt_index}_{num_inference_steps}_{i}.png")
        print(f"Image quality: {brisk(image)}")
    current_prompt_index +=1
    
    if current_prompt_index == len(story_prompts):
        break

    prompt_text = story_prompts[current_prompt_index]


    prompt_text = prompt_text + f" Child Storybook illustration, 4k"
    
    if story_prompts_polarity[current_prompt_index] != "neutral" and story_prompts_polarity[current_prompt_index] != "resilience":
        prompt_text += f", {story_prompts_polarity[current_prompt_index]}"
    
    images += image_batch.images

# Possible idea: aggregate polarity