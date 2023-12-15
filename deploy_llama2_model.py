###################################################################################
# LLM Model used: https://huggingface.co/meta-llama/Llama-2-13b-chat
# To download a model
# Step 1. Visit https://huggingface.co/meta-llama/Llama-2-13b-chat
# Step 2. git lfs install
# Step 3. git clone https://huggingface.co/meta-llama/Llama-2-13b-chat
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
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
warnings.filterwarnings("ignore")

# To provide access to multi large GPUs, since the model llama2-13b-chat is Huge
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

# num_replicas=2, if we want to create two replicas of the hosted model,
# to handle multiple request at once.
@serve.deployment(route_prefix="/llama2-13b-chat", ray_actor_options={"num_gpus": 2}, num_replicas=1)
@serve.ingress(app)
class Llama2Deployment:
    def __init__(self):
        self.ques_template = """
        [INST]<<SYS>>
        you are a helpful, respectful and honest assistant. Always answer as helpfully as possible. while being safe, please ensure that your responses are socially unbaised and positive in nature. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
        Answer the following question: {question}
        [/INST]
        """
        self.prompt_ques_template = """
        [INST]<<SYS>>
        {prompt}
        <</SYS>>
        Answer the following question: {question}
        [/INST]
        """
        self.context_ques_template = """
        [INST]<<SYS>>
        you are a helpful, respectful and honest assistant. Always answer as helpfully as possible. while being safe, please ensure that your responses are socially unbaised and positive in nature. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
        Given the context: {context}
        Answer the following question: {question}
        [/INST]
        """
        self.prompt_context_ques_template = """
        [INST]<<SYS>>
        {prompt}
        <</SYS>>
        Given the context: {context}
        Answer the following question: {question}
        [/INST]
        """

        self.max_memory_mapping = {0: "40GB", 1: "40GB"}
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=True,# For 8 bit quantization
            max_memory=self.max_memory_mapping
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune", backend="inductor")
        #self.stopping_criteria = StoppingCriteriaList([StopTokens(["</s>", "<pad>"])])

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
        generated_answer = response_query.replace("</s>", "")
        answer_with_prompt = generated_answer.strip("[/INST]")
        final_answer = answer_with_prompt[-1].strip()
        json_llm_result = {
            "query": user_query,
            "llm_answer": final_answer,
            params: params
        }
        torch.cuda.empty_cache()
        gc.collect()
        return json_llm_result

    @app.post("/health")
    def health(self):
        return {"health": "OK"}

deployment = Llama2Deployment.bind()
