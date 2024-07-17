import asyncio
import json
from datetime import datetime

from decouple import config
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Path, Query, HTTPException

from django.forms.models import model_to_dict
from fastapi import APIRouter, UploadFile, Form
from starlette.responses import FileResponse

from app.api.schemas import ChatResponse, ChatRequest
from app.model.multimodal import MultiModal_Model

api_router = APIRouter()
TIME_OUT_MIN = 5
multiModal_Model = MultiModal_Model()


@api_router.get("/api/v1/health")
async def health():
    return {"status": "healthy",
            "message": "success",
            "timestamp": datetime.now()}


@api_router.post("/api/v1/chat", response_model=ChatResponse, status_code=200)
async def chat(data: ChatRequest):
    # build ChatResponse class and return
    text_input = data.message
    response = multiModal_Model.generator(text_input)
    print(response)
    chat_response = ChatResponse(message=response['text'], images=[response['image']])
    return chat_response


@api_router.get("/api/v1/images/{image_id}")
async def get_image(image_id: str):
    try:
        # read image path and return file response
        file_dir = f"data/{image_id}"
        return FileResponse(file_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
