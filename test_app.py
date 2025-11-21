from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr, HttpUrl
import uvicorn

from scripts import s3
import os
import time

import torch
from transformers import pipeline
from transformers import AutoImageProcessor

app = FastAPI()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

######## DOWNLOAD ML MODELS ##########

force_download = False

model_name = 'tinybert-sentiment-analysis'
local_path = os.path.join('ml-models', model_name)
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
sentiment_classifier = pipeline('text-classification', model=local_path, device=device)



model_name = 'tinybert-disaster-tweet'
local_path = os.path.join('ml-models', model_name)
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
tweet_classifier = pipeline('text-classification', model=local_path, device=device)

model_name = 'vit-human-pose-classification'
local_path = os.path.join('ml-models', model_name)
if not os.path.isdir(local_path) or force_download:
    s3.download_dir(local_path, model_name)
pose_classifier = pipeline('image-classification', model=local_path, device=device,image_processor=image_processor)

######## DOWNLOAD ENDS  #############


@app.get("/")
def read_root():
    return "Hello! I am up!!!"


@app.post("/app/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_classifier(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-sentiment-analysis",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)

    return output

@app.post("/app/v1/disaster_analysis")
def disaster_analysis(data: NLPDataInput):
    start = time.time()
    output = tweet_classifier(data.text)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)
    
    return output

@app.post("/app/v1/pose_analysis")
def pose_analysis(data: ImageDataInput):
    start = time.time()
    # print(data)
    urls = [str(x) for x in data.url]
    output = pose_classifier(urls)
    end = time.time()
    prediction_time = int((end-start)*1000)

    labels = [x['label'] for x in output[0]]
    scores = [x['score'] for x in output[0]]

    output = ImageDataOutput(model_name="vit-human-pose-classification",
                           url = data.url,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)
    
    return output



if __name__=="__main__":
    uvicorn.run(app = "test_app:app", port=8001,reload=True)