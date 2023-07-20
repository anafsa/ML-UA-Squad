from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import os
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import re
import pickle
import glob

# story = pickle.load("stories/1689788440.pkl")
main_id = 1689788440
with open(f'stories/{main_id}.pkl', 'rb') as f:
    data = pickle.load(f)

# print(story.keys())
# print(story["history"])

for paragph in data["story"]:
    print(paragph)

story_prompts = ["Emily's Discovery: Generate an image that depicts young Emily immersed in a world of books, surrounded by shelves filled with stories that captivate her imagination and offer solace during her challenging times.",
                 "The Medical Journey: Create an image that portrays Emily's resilience and determination as she undergoes medical treatments, showcasing her bravery in the face of adversity and the unwavering support of her parents and medical professionals.",
                "A Ray of Hope: Generate an image that symbolizes Emily's newfound hope and optimism. It could feature a vibrant sunrise or a blossoming flower, representing the turning point in her battle against the disease."]

reduced_story_prompts = [re.split(r'[,\.]', story_prompt)[0] for story_prompt in story_prompts]
# print(reduced_story_prompts)

init_prompts = [reduced_story_prompt.split(":")[0] for reduced_story_prompt in reduced_story_prompts]
# print(init_prompts)

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
prompt_text = f"{story_prompts[current_prompt_index]} Child Storybook illustration, 4k"# by Matt Bors"

previous_prompt = ""
if current_prompt_index > 0:
    previous_prompt = reduced_story_prompts[current_prompt_index-1]
    prompt_text = prompt_text + f". Generate an image as a storybook illustration by Matt Bors"


images = []

story_folder = "ES_8"
os.makedirs(story_folder, exist_ok = True)

print(f"*** Generating {total_num_images} image(s) ***")
for i in range(len(story_prompts)):
    print(f"* Batch {i + 1} of {num_batches} *")

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
        image_batch.images[i].save(f"{story_folder}/img_{current_prompt_index}_{num_inference_steps}_{i}.png")
    current_prompt_index +=1
    
    if current_prompt_index == len(story_prompts):
        break

    prompt_text = story_prompts[current_prompt_index]

    prompt_text = prompt_text + f" Child Storybook illustration, 4k "# by Matt Bors"
    images += image_batch.images

# Possible paths: mixing of two consecutive images.