from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import os
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
from uuid import uuid4

DEVICE_CPU = torch.device('cpu:0')
DEVICE_GPU_0 = torch.device('cuda:0')
DEVICE_GPU_1 = torch.device('cuda:1')

# Loading encoder and prior pipeline into the RAM to be run on the CPU
# and unet and decoder to the VRAM to be run on the GPU.
# Note the usage of float32 for the CPU and float16 (half) for the GPU
# Set the `local_files_only` to True after the initial downloading
# to allow offline use (without active Internet connection)
print("*** Loading encoder ***")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    subfolder='image_encoder',
    cache_dir='./kand22'
).to(DEVICE_CPU)

print("*** Loading unet ***")
unet = UNet2DConditionModel.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    subfolder='unet',
    cache_dir='./kand22'
).half().to(DEVICE_GPU_0)

print("*** Loading prior ***")
prior = KandinskyV22PriorPipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-prior',
    image_encoder=image_encoder, 
    torch_dtype=torch.float32,
    cache_dir='./kand22'
).to(DEVICE_CPU)

print("*** Loading decoder ***")
decoder = KandinskyV22Pipeline.from_pretrained(
    'kandinsky-community/kandinsky-2-2-decoder',
    unet=unet,
    torch_dtype=torch.float16,
    cache_dir='./kand22'
).to(DEVICE_GPU_0)


job_id = str(uuid4())

num_batches = 1
images_per_batch = 1
total_num_images = images_per_batch * num_batches

negative_prior_prompt = 'worst quality, low quality'

prompt_text = "Emily's Discovery: Generate an image that depicts young Emily immersed in a world of books."

prompt_text = "Generate an image depicting The Medical Journey of young Emily, \
    incorporating elements from the image generated using this prompt: {prompt_text}. \
    Capture the resilience and determination of Emily as she undergoes medical treatments, \
    while maintaining the vibrant and immersive world of books that symbolizes her escape and source of hope during challenging times."

images = []

print(f"*** Generating {total_num_images} image(s) ***")
for i in range(num_batches):
    print(f"* Batch {i + 1} of {num_batches} *")
    
    # Generating embeddings on the CPU
    img_emb = prior(
        prompt=prompt_text,
        num_inference_steps=50,
        num_images_per_prompt=images_per_batch)

    negative_emb = prior(
        prompt=negative_prior_prompt,
        num_inference_steps=50,
        num_images_per_prompt=images_per_batch
    )

    # Converting fp32 to fp16, to run decoder on the GPU
    image_batch = decoder(
        image_embeds=img_emb.image_embeds,
        negative_image_embeds=negative_emb.image_embeds,
        num_inference_steps=50, height=512, width=512)

    images += image_batch.images

# Saving the images
os.mkdir(job_id)
for (idx, img) in enumerate(images):
    img.save(f"{job_id}/img_{job_id}_{idx + 1}.png")