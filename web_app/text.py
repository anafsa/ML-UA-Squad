import re
import time
import requests
import json
import pickle
import openai

# OpenAI API credentials
AZURE_OPENAI_KEY = "41d471bedea9412e9d0a5461cf69bfc8"
AZURE_OPENAI_ENDPOINT = "https://openai-resource-team-6-france.openai.azure.com/"
AZURE_ENGINE_NAME = "gpt35-team-6"

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = AZURE_OPENAI_KEY


def ask_gpt(command, text, temp=0):
    '''
    Ask the model to generate text based on the command and the text provided.

    Parameters:
        command (str): The command to be used.
        text (str): The text to be used.
    '''
    time.sleep(1)  # to avoid Azure API throttling
    content = openai.ChatCompletion.create(
        engine=AZURE_ENGINE_NAME,
        messages=[
            {"role": "system", "content": command},
            {"role": "user", "content": text}
        ],
        temperature=temp,
    )
    return content.choices[0].message["content"]


def get_story(request):
    '''
    Get a story based on the request.

    Parameters:
        request (str): The request from the user.
    '''
    command = "You are a helpful storyteller for children."
    return ask_gpt(command, request)


def get_summary(story):
    '''
    Get a summary of the story.

    Parameters:
        story (str): The story to be summarized.
    '''
    command = "You will be provided with a story, and your task is to summarize the story using at most 70 words."
    return ask_gpt(command, story)


def get_prompt_to_generate_image(story):
    '''
    Get a prompt to generate an image based on the story.

    Parameters:
        story (str): The story to be used.
    '''
    command = "Can you give me one prompt to generate one image based on this story in stable diffusion?"
    response = ask_gpt(command, story)
    try:
        rp = response.split(':')[1]
        return rp
    except:
        return response


def get_title(story, attempt=0):
    command = "You will be provided with a story, and your task is to create a title for it using at most 15 words."
    title = ask_gpt(command, story)
    num_words = len(title.split())
    if (num_words > 15 or num_words == 0):
        if attempt < 3:
            return get_title(story, attempt+1)
        else:
            return "Once upon a time..."
    return title


def get_feeling(story, attempt=0):
    command = "You will be provided with a short story. Your task is to summarize the story using one word that describes the overall emotion " +\
        "of the paragraph. For example, if the story is about a happy event, you might use the word joy to summarize it. You should use one of " +\
        "the following words: happiness, sadness, anger, fear, surprise, disgust, contempt, resilience."
    feeling = ask_gpt(command, story).lower()
    if feeling not in ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "contempt", "resilience", "neutral"]:
        if attempt < 3:
            return get_feeling(story, attempt+1)
        else:
            return "neutral"
    return feeling


def get_sentiment_level(story, attempt=0):
    command = "You will be provided with a short story. Your task is to get the sentiment level of the overall paragraph. " +\
        "You should use one of the following words: very positive, positive, neutral, negative, very negative."
    sentiment = ask_gpt(command, story).lower()
    if sentiment not in ["very positive", "positive", "neutral", "negative", "very negative"]:
        if attempt < 3:
            return get_sentiment_level(story, attempt+1)
        else:
            return "neutral"
    return sentiment


def get_tags(story, attempt=0):
    command = "You will be provided with a short story. Your task is to get the categories of the overall paragraph. You should use three of" +\
        " the following words: action, adventure, animation, biography, comedy, crime, documentary, drama, family, fantasy, history," +\
        " horror, music, musical, mystery, romance, sci-fi, short, sport, thriller, war, western."
    tags = ask_gpt(command, story).lower()
    tags = tags.split(", ")

    valid_tags = ["action", "adventure", "animation", "biography", "comedy", "crime", "documentary", "drama", "family", "fantasy", "history",
                  "horror", "music", "musical", "mystery", "romance", "sci-fi", "short", "sport", "thriller", "war", "western"]

    # verify if list contains only 3 elements, and if each element is a valid tag
    if len(tags) != 3 or not all(elem in valid_tags for elem in tags):
        if attempt < 3:
            return get_tags(story, attempt+1)
        else:
            return ["family", "short", "animation"]
    return tags


def get_paragraphs(story):
    # split output_chatgpt into paragraphs
    paragraphs = story.split('\n\n')
    paragraphs = [x for x in paragraphs if x != '']
    # remove spaces from the beginning of each paragraph
    paragraphs = [x[1:] if x[0] == ' ' else x for x in paragraphs]
    return paragraphs


def get_scenes(story):
    scenes = []

    paragraphs = get_paragraphs(story)

    for idx, paragraph in enumerate(paragraphs):
        scene = {
            "id": str(idx),
            "text": paragraph,
            "sentiment": get_sentiment_level(paragraph),
            "polarity": get_feeling(paragraph),
            "image_prompt": get_prompt_to_generate_image(paragraph),
            "tone_song": None,
            "image": None
        }

        scenes.append(scene)

    return scenes


def create_story(request, story=None):
    '''
    The main function to create a story.

    Parameters:
        request (str): The request from the user.
        story (str): The story to be used. If None, a story will be generated.
    '''
    print("Starting story generation with: " + request)
    start = time.time()

    # For this case, the main_id is the timestamp when the story is created
    main_id = str(int(time.time()))

    if story is None:
        story = get_story(request)

    summ = get_summary(story)
    title = get_title(story)

    main_story = {
        "id": f"{main_id}_{title.replace(' ', '_').replace(':', '')}",
        "title": title,
        "summary": summ,
        "len_summary": len(summ.split()),
        "len_story": len(story.split()),
        "main_prompt": request,
        "tone": get_feeling(story),
        "tags": get_tags(story),
        "cover_image_prompt": get_prompt_to_generate_image(summ),
        "story": get_scenes(story),
    }

    end = time.time()

    main_story["elapsed_time"] = end - start
    print(f"Main story: {main_story['elapsed_time']}")
    return main_story


def save_story(main):
    '''
    Save the story to a binary file (pickle) and a json file.
    '''

    # convert to pickle
    main_id = main["id"]
    with open(f'stories/{main_id}.pkl', 'wb') as f:
        pickle.dump(main, f)

    # convert to json
    main_json = json.dumps(main)
    # save to file
    with open(f'stories/{main_id}.json', 'w') as f:
        f.write(main_json)
    print("Saved story: " + main_id)


if __name__ == "__main__":
    request = "What is an odontologist?"
    main = create_story(request)
    save_story(main)
    print(main)
