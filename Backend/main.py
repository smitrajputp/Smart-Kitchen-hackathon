#changed:
# in input_data_chat class, 'input' and 'image' input types now can be none and new key val pair of 'location' is added.

import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from agent import ai_ImgAnalyser ,ai_InventoryManeger ,direct_image, pdf_to_knowledgebase, connect_milvus
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials
from dotenv import load_dotenv


from contextlib import asynccontextmanager
from pymilvus import connections

@asynccontextmanager
async def lifespan(app: FastAPI):
    connect_milvus()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class input_data_ai_ImgAnalyser(BaseModel):
    image: str|None
    location: str|None

class input_data_ai_InventoryManeger(BaseModel):
    input: str|None
    location: str|None

class input_data_vision(BaseModel):
    image: str
    location: str

class input_knowlwedgebase(BaseModel):
    path: str

@app.get("/")
async def confirmation():
    return {"message":"API working!"}


@app.post("/ai_ImgAnalyser")
async def model1(input: input_data_ai_ImgAnalyser):
    response = ai_ImgAnalyser(input)
    try:
        # Extract the JSON portion from the response string
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_part = response[start_idx:end_idx + 1]
            # Convert JSON string to a dictionary
            meta_data = json.loads(json_part)
            # Remove the JSON part from the original response
            response_without_json = response.replace(json_part, '').strip()
            # Use rpartition to split at the last occurrence of '\n'
            response_without_json, _, _ = response_without_json.rpartition('\n')
        else:
            raise ValueError("No valid JSON found in the response.")

        return {
            "response": response_without_json,
            "meta_data": meta_data
        }
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")
    # return {"response": response}    


@app.post("/direct_img")
async def model2(input: input_data_vision):
    response = direct_image(input)
    try:
        # Extract the JSON portion from the response string
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_part = response[start_idx:end_idx + 1]
            # Convert JSON string to a dictionary
            meta_data = json.loads(json_part)
            # Remove the JSON part from the original response
            response_without_json = response.replace(json_part, '').strip()
            # Use rpartition to split at the last occurrence of '\n'
            response_without_json, _, _ = response_without_json.rpartition('\n')
        else:
            raise ValueError("No valid JSON found in the response.")

        return {
            "response": response_without_json,
            "meta_data": meta_data
        }
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")
    # return {"response": response}

@app.post("/ai_InventoryManeger")
async def model3(input: input_data_ai_InventoryManeger):
    response = ai_InventoryManeger(input)
    return {"response": response}

@app.post("/knowledgebase")
async def model4(input: input_knowlwedgebase):
    response = pdf_to_knowledgebase(input)
    return {"response": response}