###################################################################################
# LLM Model used: https://huggingface.co/google/flan-ul2
# To download a model
# Step 1. Visit https://huggingface.co/google/flan-ul2
# Step 2. git lfs install
# Step 3. git clone https://huggingface.co/google/flan-ul2
# Developer: Shobhit Agarwal
####################################################################################
import gc
import os
import sys
import copy
import torch
import warnings
from ray import serve
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Union
from sentence_transformers import CrossEncoder
from accelerate import infer_auto_device_map
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
warnings.filterwarnings("ignore")

# To provide access to multi large GPUs, since the model flan-ul2 is Huge
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

LLM_MODEL_PATH = "<model_path>"

if os.path.exists(LLM_MODEL_PATH):
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

class LLMParameters(BaseModel):
    top_k: int = 1
    top_n: float = 1.0
    temperature: float = 0.2
    repetition_penalty: float = 1.4
    max_new_tokens: int = 512
    min_new_tokens: int = 64

class LLMQuery(BaseModel):
    prompt: Union[str, None] = ""
    context: Union[str, None] =  ""
    query: Union[str, None] = ""
    params: Union[LLMParameters, None] = None

class StopTokens(StoppingCriteria):

    def __init__(self, stop_ids: list) -> None:
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores: torch.FloatTensor,
                 **kwargs) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
            else:
                return False

@serve.deployment(route_prefix="/flan-ul2", ray_actor_options={"num_gpus": 2})
@serve.ingress(app)
class FlanDeployment:
    def __init__(self):
        self.ques_template = """
        you are a helpful, respectful and honest assistant. Always answer as helpfully as possible. while being safe, please ensure that your responses are socially unbaised and positive in nature. \n\n
        Answer the following question: {question}
        """
        self.prompt_ques_template = """
        {prompt}\n\n
        Answer the following question: {question}
        """
        self.context_ques_template = """
        you are a helpful, respectful and honest assistant. Always answer as helpfully as possible. while being safe, please ensure that your responses are socially unbaised and positive in nature. \n\n
        Given the context: {context}\n\n
        Answer the following question: {question}
        """
        self.prompt_context_ques_template = """
        {prompt}\n\n
        Given the context: {context}\n\n
        Answer the following question: {question}
        """

        self.max_memory_mapping = {0: "40GB", 1: "40GB"}
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,# For 8 bit quantization
            max_memory=self.max_memory_mapping
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune", backend="inductor")
        self.stopping_criteria = StoppingCriteriaList([StopTokens(["</s>", "<pad>"])])

    @app.post("/ask_query")
    def ask_query(self, request: LLMQuery):
        params = LLMParameters() if request.params is None else request.params
        user_query = request.query
        user_prompt = request.prompt
        user_context = request.context

        if user_context == "" or user_context == None:
            # User not passing neither context nor prompt
            if user_prompt == "" or user_prompt == None:
                prompt = copy.deepcopy(self.ques_template)
                input_text = prompt.format(question=user_query)
            # User not passing context but passing prompt
            else:
                prompt = copy.deepcopy(self.prompt_ques_template)
                input_text = prompt.format(prompt=user_prompt, question=user_query)
        else:
            # User passing context but no prompt
            if user_prompt == "" or user_prompt == None:
                prompt = copy.deepcopy(self.context_ques_template)
                input_text = prompt.format(context=user_context, question=user_query)
            # User passing context and also the prompt
            else:
                prompt = copy.deepcopy(self.prompt_context_ques_template)
                input_text = prompt.format(prompt=user_prompt, context=user_context, question=user_query)

        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        tokens = self.model.generate(
            **input_ids,
            max_new_tokens=params.max_new_tokens,
            min_new_tokens=params.min_new_tokens,
            do_sample=True,
            temperature=params.temperature,
            top_k=params.top_k,
            top_n=params.top_n,
            repetition_penalty=params.repetition_penalty,
            stopping_criteria=self.stopping_criteria
        )
        response_query = self.tokenizer.decode(tokens[0])
        generated_answer = response_query.replace("<pad>", "").replace("</s>", "")
        generated_answer = generated_answer.strip()
        json_llm_result = {
            "query": user_query,
            "llm_answer": generated_answer,
            params: params
        }
        torch.cuda.empty_cache()
        gc.collect()
        return json_llm_result

    @app.post("/health")
    def health(self):
        return {"health": "OK"}


deployment = FlanDeployment.bind()
