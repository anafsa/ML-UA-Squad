from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
import os
from transformers import CLIPVisionModelWithProjection
from diffusers.models import UNet2DConditionModel
import re
import pickle
import glob
from pyiqa import create_metric
import argparse
import numpy as np
import time


def brisk(image):
    metric_name = "brisque"
    iqa_model = create_metric(metric_name, metric_mode="NR")
    score = iqa_model(image)
    return score.item()


def brisk_for_multiple_images(images):
    metric_name = "brisque"
    iqa_model = create_metric(metric_name, metric_mode="NR")
    scores = []
    for image in images:
        score = iqa_model(image)
        scores.append(score.item())
    return scores


def multiple_images_evaluation(images, metric_name="clipiqa+"):
    iqa_model = create_metric(metric_name, metric_mode="NR")
    scores = []
    for image in images:
        score = iqa_model(image)
        scores.append(score.item())
    return scores


def generate_image(args):
    start = time.time()

    file_path = args.input_story
    if file_path == default_story_path:
        print("Choosen story is the default story")
        print("Make sure that you specified the story")

    # if file_path.startswith("stories") == False:
    #     file_path = f"stories/{file_path}"

    if file_path.endswith(".pkl") == False:
        file_path = f"{file_path}.pkl"

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    story_folder_path = re.split(r'[\/\.]', file_path)[-2]

    # story_folder_path = "Default"
    # for file_path in glob.glob(r"stories/*.pkl"):
    #     story_folder_path = re.split(r'[\/\.]', file_path)[1]
    #     with open(file_path, 'rb') as f:
    #         data = pickle.load(f)

    print("Story folder: ", story_folder_path)

    story_prompts = []
    story_title_prompt = data["cover_image_prompt"]
    divided_story_title_prompt = story_title_prompt.split(":")
    if len(divided_story_title_prompt) > 1:
        story_title_prompt = divided_story_title_prompt[1]

    story_prompts.append(story_title_prompt)

    story_prompts_polarity = ["neutral"]
    for i, story_paragraph_data in enumerate(data["story"]):
        # paragh_story = story_paragraph_data["text"]
        # print(f"Paragh {i+1}: {paragh_story}")
        divided_image_prompt = story_paragraph_data["image_prompt"].split(":")
        if len(divided_image_prompt) > 1:
            story_prompts.append(divided_image_prompt[1])
        else:
            story_prompts.append(story_paragraph_data["image_prompt"])
        story_prompts_polarity.append(story_paragraph_data["polarity"])

    print(f"Cover Image prompt: {story_title_prompt}")

    print(f"Story_prompts: {story_prompts}")

    DEVICE_CPU = torch.device('cpu:0')
    DEVICE_GPU_0 = torch.device('cuda:0')
    DEVICE_GPU_1 = torch.device('cuda:1')

    print("*** Loading encoder ***")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        subfolder='image_encoder',
        cache_dir='./kand22',
        local_files_only=True
    ).to(DEVICE_CPU)

    print("*** Loading unet ***")
    unet = UNet2DConditionModel.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        subfolder='unet',
        cache_dir='./kand22',
        local_files_only=True
    ).half().to(DEVICE_GPU_0)

    print("*** Loading prior ***")
    prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        image_encoder=image_encoder,
        torch_dtype=torch.float32,
        cache_dir='./kand22',
        local_files_only=True
    ).to(DEVICE_CPU)

    print("*** Loading decoder ***")
    decoder = KandinskyV22Pipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        unet=unet,
        torch_dtype=torch.float16,
        cache_dir='./kand22',
        local_files_only=True
    ).to(DEVICE_GPU_0)

    images_per_batch = args.batch_size

    negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, \
        duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, \
            mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured"

    current_prompt_index = 0
    prompt_text = f"{story_prompts[current_prompt_index]} Child Storybook illustration, 4k"

    images = []

    os.makedirs("images/" + story_folder_path, exist_ok=True)
    # extended_story_folder_path = story_folder_path
    extended_story_folder_path = f"images/{story_folder_path}/all"
    if images_per_batch > 1:
        os.makedirs(extended_story_folder_path, exist_ok=True)

    for i in range(len(story_prompts)):
        print(f"* Paragraph {i} of {len(story_prompts)- 1} *")

        print(f"Prompt: \n{prompt_text}")

        num_inference_steps = 50
        bad_image_generated = True

        while bad_image_generated:
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

            maniqa_scores = multiple_images_evaluation(
                image_batch.images, metric_name="maniqa")
            clip_scores = multiple_images_evaluation(
                image_batch.images, metric_name="clipiqa+")
            assembled_scores = list(np.multiply(maniqa_scores, clip_scores))
            max_image_score = max(assembled_scores)
            max_value_image = image_batch.images[assembled_scores.index(
                max_image_score)]
            if max_image_score > args.threshold:
                bad_image_generated = False
            else:
                print("Retrying image generation")
                print(f"Max assembled score: {max_image_score:.2f}")

            for i, image in enumerate(image_batch.images):
                image_batch.images[i].save(
                    f"{extended_story_folder_path}/img_{current_prompt_index}_{clip_scores[i]:.2f}_{maniqa_scores[i]:.2f}_{assembled_scores[i]:.2f}_{i}.png")

        max_value_image.save(
            f"images/{story_folder_path}/img_{current_prompt_index}_{max_image_score:.2f}.png")
        current_prompt_index += 1

        if current_prompt_index == len(story_prompts):
            break

        prompt_text = story_prompts[current_prompt_index]

        prompt_text = prompt_text + f" Child Storybook illustration, 4k"

        # if story_prompts_polarity[current_prompt_index] != "neutral" and story_prompts_polarity[current_prompt_index] != "resilience":
        #     prompt_text += f", {story_prompts_polarity[current_prompt_index]}"
    end = time.time()
    dtime = end - start
    return {"time": dtime, "bs": images_per_batch}


    # default_story_path = "stories/1689861518_Safe_and_Playful_Timmy's_Allergy_Adventure.pkl"
    # default_story_path = "stories/1689942495_The_Magical_Camera_Lily's_Journey_Inside_the_MRI_Machine.pkl"
default_story_path = "stories/1690106398_Tooth_Sleuths_Unraveling_Mysteries_and_Identifying_Individuals_through_Dental_Clues.pkl"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_story', type=str,
                        default=default_story_path, help='input image/folder path.')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help='number of images to be generated per paragph.')
    parser.add_argument('-t', '--threshold', type=float, default=0.2,
                        help='Threshold for image quality generation.')
    args = parser.parse_args()
    # args["input_story"] = "path"
    args.batch_size = 3
    args.threshold = .2
    generate_image(args)
