import os
import requests
from datetime import datetime

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI 
import google.generativeai as genai

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_community.tools.tavily_search import TavilySearchResults

import base64
import httpx

import firebase_admin
from firebase_admin import credentials, db


# Replace '\\n' with '\n' in the private key
private_key = os.environ.get('PRIVATE_KEY').replace('\\n', '\n')

DATA={
        "type": "service_account",
        "project_id": os.getenv("project_id"),
        "private_key_id": os.getenv("private_key_id"),
        "private_key": private_key,
        "client_email": os.getenv("client_email"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "client_id": os.getenv("client_id"),
        "token_uri": "https://accounts.google.com/o/oauth2/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("client_x509_cert_url"),
        "universe_domain": os.getenv("universe_domain")
    }

cred = credentials.Certificate(DATA)
firebase_admin.initialize_app(cred, {'databaseURL': 'https://foodai-7ebf0-default-rtdb.firebaseio.com/'})

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

web_search_tool = TavilySearchResults()
web_search_tool.name = "Web_Search"
web_search_tool.description = "Retrieve relevant info from web."

class web_search_inputs(BaseModel):
   query: str = Field(description="query for searching on web")

web_search_tool.args_schema = web_search_inputs

def get_current_weather(city_name):
    if not WEATHER_API_KEY:
        raise ValueError("WEATHER_API_KEY environment variable is not set.")
    
    # API URL
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric"
    
    # API request
    response = requests.get(url)
    
    # Raise an exception if the request fails
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code}, {response.text}")
    
    # Parse and return the JSON response
    return response.json()

@tool
def visual_tool(prompt: str, image_url: str, loc: str) -> str:
    """Responding on image url input and location
    Args:
        prompt (str): prompt input
        image_url (str): url input
        loc (str): location input
    """
    # Get current date and time
    now = datetime.now()
    user_city = loc #only city
    weather_api_res = get_current_weather(user_city)
    temp_and_humi = f"Temperature: {weather_api_res['main']['temp']}°C\nHumidity: {weather_api_res['main']['humidity']}%"
    response = requests.get(image_url)
    if response.status_code == 200:
        pass
    else:
        return(f"Error fetching image. Status code: {response.status_code}")

    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt+"\n"+str(now)+"\n"+temp_and_humi},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    try:
        response = model.invoke([message])
        print("Respond succesfully: ", response.content)
        return response.content
    except Exception as e: 
        print("Error found : ",e)
        return e

visual_tool.name = "visual_tool"
visual_tool.description = "Gets image details on given prompt"

class visual_inputs(BaseModel):
   prompt: str = Field(description="prompt for image")
   image_url: str = Field(description="url for image")
   loc: str = Field(description="geographical location of the image")
visual_tool.args_schema = visual_inputs

with open('DatabaseAI_prompt.txt', 'r') as file:
    SystemMessage = file.read()

@tool
def database_tool(Agent_prompt: str) -> str:
    """Responding on user's database query"""

    message = SystemMessage+"\n"+f"user: {Agent_prompt}."
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    try:
        command = model.invoke(message)
        print("LLM Responded Successfully: ", command.content)
    except Exception as e:
        print("Error invoking LLM: ", e)
        return {"Error": "Failed to process the prompt with LLM.", "Details": str(e)}
    
    command_content = command.content.strip().lower()  # Remove any leading/trailing whitespace
    print(command_content)
    # List of valid categories based on your database structure
    valid_categories = ["dairy", "fruits", "vegetables", "others"]
    
    # Check if the command is not "None" and is a valid path
    if command_content and command_content != "none":
        if command_content == "/":
            # Handle root command to fetch all data
            try:
                ref = db.reference("/")  # Reference to the root of the database
                response = ref.get()  # Get all data from the database
                print("All Firebase Data Fetched Successfully: ", response)
                return response
            except Exception as e:
                print("Error Fetching All Data from Firebase: ", e)
                return {"Error": "Failed to fetch all data from Firebase.", "Details": str(e)}
        elif any(command_content.startswith("/"+category) for category in valid_categories):
            try:
                ref = db.reference(command_content)
                response = ref.get()  # .get() retrieves the data from the database
                print("Firebase Data Fetched Successfully: ", response)
                return response
            except Exception as e:
                print("Error Fetching Data from Firebase: ", e)
                return {"Error": "Failed to fetch data from Firebase.", "Details": str(e)}
        else:
            return {"Error": "Invalid command structure returned from LLM."}
    else:
        # If LLM returns "None" or empty
        return {"Response": "Not present in the database."}
    

database_tool.name = "database_tool"
database_tool.description = "gets all the inventory item related information according to the given prompt."

class database_inputs(BaseModel):
   Agent_prompt: str = Field(description="prompt for retreving inventory related information fom the database.")

database_tool.args_schema = database_inputs

@tool
def knowledgebase_tool(Agent_query: str) ->str:
    # Get embedding of query
    query_embedding = get_embedding(Agent_query)

    # Search similar content in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding], anns_field="embedding", param=search_params, limit=3, output_fields=["text"]
    )

    # Retrieve most relevant context
    context = " ".join([hit.entity.get("text") for hit in results[0]])

    # Query Gemini with context
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Use this context:\n{context}\n\nAnswer this: {Agent_query}")

    return {"response": response.text}


def get_embedding(text: str) -> list:
    """
    Converts text into an embedding vector using Gemini.
    """
    model = genai.GenerativeModel("gemini-embed")  # Use embedding model
    embedding = model.generate_content(text)
    return embedding.text  # Convert text embedding into float vector

knowledgebase_tool.name = "knowledgebase_tool"
knowledgebase_tool.description = "gets information useful for making prediction from the custom knowledgebase provided by developer."

class knowledgebase_inputs(BaseModel):
   Agent_prompt: str = Field(description="prompt for retreving custom knowledge from the milvus vector db, useful in making predictions.")

database_tool.args_schema = knowledgebase_inputs

TOOLS = [web_search_tool, visual_tool, database_tool, knowledgebase_tool]