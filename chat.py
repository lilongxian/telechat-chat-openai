import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
from cc.base import ChatCompletionResponse, ChatCompletionRequest
from cc.util import predict_stream
import time
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
import gc
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple
from cc.api import main


if __name__ == '__main__':
    main()

