from typing import Union
from fastapi import FastAPI
from starlette.responses import FileResponse
import argparse
from text import create_story, save_story
from image import generate_image
import os
import glob
import re
import json
from os.path import exists


def dict_to_namespace(input_dict):
    namespace = argparse.Namespace(**input_dict)
    return namespace


app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


@app.get("/")
def read_index():
    return FileResponse('index.html')


@app.get("/scene.html")
def read_index():
    return FileResponse('scene.html')

@app.get("/profile.html")
def read_index():
    return FileResponse('profile.html')

@app.get("/{folder}/{file_name}")
def read_src(folder, file_name):
    return FileResponse(f'{folder}/{file_name}')


@app.get("/{folder}/{sub}/{file_name}")
def read_src(folder, sub, file_name):
    return FileResponse(f'{folder}/{sub}/{file_name}')


@app.get("/generate/")
def createStory(request: str = "cars and heroes"):

    ss = create_story(request)
    save_story(ss)
    payload = json.load(open("stories/" + ss["id"] + ".json"))

    # payload = json.load(
    #     open("stories/1690128944_Oliver's_Redemption_A_Mischievous_Adventure_of_Greed,_Lessons,_and_Transformation.json"))

    generateFinalStory(payload)


def generateFinalStory(payload):
    args = {}
    args["input_story"] = "stories/" + payload["id"].replace(":", "") + ".pkl"
    args["batch_size"] = 3
    args["threshold"] = .2

    img_dict = {"time": None, "bs": 1}

    if (not exists("images/" + payload["id"].replace(":", ""))):
        img_dict = generate_image(dict_to_namespace(args))

    payload["image_generation_time"] = img_dict["time"]
    payload["batch_size"] = img_dict["bs"]
    basePath = f"images/{payload['id'].replace(':', '')}/"
    allFiles = os.listdir(basePath)
    payload["thumbnail"] = basePath + [
        i_path for i_path in allFiles if i_path.startswith('img_0')][0]

    for i in range(len(payload["story"])):
        img_path = [
            i_path for i_path in allFiles if i_path.startswith('img_'+str(i+1))][0]
        payload["story"][i]["image"] = basePath + img_path

    json.dump(payload, open("finalStories/" +
              payload['id'].replace(':', '') + ".json", 'w'), indent=4)
    print("Payload Saved, ID: " + payload['id'].replace(':', ''))


def createStories():
    with open('prompts.txt') as f:
        for prompt in f.read().split("\n"):
            createStory(prompt)
    # for file_path in glob.glob(r"../stories/*.json"):
    #     story_folder_path = re.split(r'[\/\.]', file_path)[1]
    #     with open(file_path, 'rb') as f:
    #         data = pickle.load(f)

    # {"img": , "title": , "json_path":}
    # {"img": , "title": , "json_path":}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    # createStories()
    # print(os.listdir("finalStories"))
    for p in os.listdir("finalStories"):
        # with () as f:
        j = json.load(open("finalStories/" + p))
        print({"img": j["thumbnail"], "title": j["title"], "json_path":p})