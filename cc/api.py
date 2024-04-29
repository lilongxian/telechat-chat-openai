
"""
CreateTime: 2024-04-20
Author: lilongxian
Description:
1. OpenAI style standard streaming multi-turn dialogue interface WEB API.
2. Telechat-12b is supported by default, you can also support Telechat-7b & Telechat-11b by modifying the tokenizer loading here.
3. This project only opens the chat-openai version of telechat. agent-chat-openai currently has no open source plans.

"""

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
from cc.base import ChatCompletionResponse, ChatCompletionRequest
from cc.util import predict_stream
from transformers import GenerationConfig
from telechat.tokenization_telechat import TelechatTokenizer as AutoTokenizer
from telechat.modeling_telechat import TelechatForCausalLM as AutoModel
from config import MODEL_TYPE, MODEL_PATH

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000
TOKENIZER_PATH = MODEL_PATH

global model, tokenizer, generate_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agent/chat", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer, generate_config

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        agent=request.agent
    )
    model_type = MODEL_TYPE.lower()
    gen_params["tools"] = []
    predict_stream_generator = predict_stream(model, tokenizer, model_type, gen_params, generate_config, model_type)
    return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")


def main():
    global model, tokenizer, generate_config
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH, device_map="auto").eval()
    generate_config = GenerationConfig.from_pretrained(MODEL_PATH)
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)


if __name__ == "__main__":

    main()

