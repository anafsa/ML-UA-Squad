from kandinsky2 import get_kandinsky2
prompt_text = "Emily's Discovery: Generate an image that depicts young Emily immersed in a world of books."
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
