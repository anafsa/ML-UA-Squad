from kandinsky2 import get_kandinsky2
prompt_text = "Emily's Discovery: Generate an image that depicts young Emily immersed in a world of books."

prompt_text = "The Medical Journey: Create an image that portrays Emily's resilience and determination as she undergoes medical treatments, \
    showcasing her bravery in the face of adversity and the unwavering support of her parents and medical professionals.\
    Generate an image depicting The Medical Journey, \
    incorporating elements from the image generated using this prompt: {prompt_text}."

model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1')
images = model.generate_text2img(
    prompt_text, 
    num_steps=50,
    batch_size=1,
    guidance_scale=4,
    h=1024,
    w=768,
    sampler='p_sampler'
)

#write image with the first three letters of the prompt
images[0].save("{0}.png".format(prompt_text.strip()[:3]))
