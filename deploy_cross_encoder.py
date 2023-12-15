###################################################################################
# Cross Encoder used is: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2
# To download a model
# Step 1. Visit https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2
# Step 2. git lfs install
# Step 3. git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-2-v2
# Developer: Shobhit Agarwal
####################################################################################
import gc
import os
import sys
import torch
import warnings
from ray import serve
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder
warnings.filterwarnings("ignore")

# To provide access to only a single GPU, since the model is very small.
# Only needed if the CPU machine is very degraded.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
XENCODER_MODEL_PATH = "<cross encoder path>"

if os.path.exists(XENCODER_MODEL_PATH):
    print("proceed! Model Exists locally")
else:
    sys.exit()

# Initialize FastAPI Application
app = FastAPI()

origins = ["*"]
app.add.middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input for Cross Encoder API
class CrossEncoderInput(BaseModel):
    sentences: List[List[str]] = [[""]]

@serve.deployment(route_prefix="/cross_encoder", ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class CrossEncoderDeployment:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model = CrossEncoder(XENCODER_MODEL_PATH, device=device)

    # Function calling prediction
    @app.post("/compare_sentences")
    def compare_sentences(self, request: CrossEncoderInput):
        sentences = request.sentences
        scores = list(self.model.predict(sentences))
        json_result = {
            "sentences": sentences,
            "predict_score": scores
        }
        # Empty cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()
        return json_result

    @app.post("/health")
    def health(self):
        return {"health": "OK"}

deployment = CrossEncoderDeployment.bind()
