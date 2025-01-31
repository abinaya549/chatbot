from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from jose import jwt
from jose.exceptions import JWTError
from datetime import datetime, timedelta
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

# Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# App Initialization
app = FastAPI(title="Chatbot Backend")

# Initialize Qdrant Client (Running Locally)
qdrant_client = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "chatbot_docs"

# Create Qdrant Collection (if not exists)
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Embedding Model (Using OpenAI)
embeddings = OpenAIEmbeddings()

# Load Vector Store
vector_store = Qdrant.from_client(qdrant_client, collection_name=COLLECTION_NAME, embeddings=embeddings)


# Models
class User(BaseModel):
    username: str
    password: str

class Query(BaseModel):
    query: str

# sample data
fake_users_db = {"admin": "password"}

# Authentication
def create_access_token(username: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    token_data = {"sub": username, "exp": expiration}
    return jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str):
    if username in fake_users_db and fake_users_db[username] == password:
        return True
    return False

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid token scheme")

    try:
        payload = jwt.decode(parts[1], SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except (JWTError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# Routes
@app.post("/auth/token")
def login(user: User):
    """
    url - http://127.0.0.1:8000/auth/token
   raw data -
     {
  "username": "admin",
  "password": "password"
    }

    output:
    {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTczODE0OTgyNH0.mhyygVsO-bqpczBrNefqXt90pyBaFPVB3NT1aDKDJmc",
    "token_type": "bearer"
    }
    """
    if not authenticate_user(user.username, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chatbot/query")
def chatbot_query(query: Query, username: str = Depends(verify_token)):    
    """
    url - http://127.0.0.1:8000/chatbot/query
   raw data -
     {
  "query": "What is your name?"
    }

    output:
    {
    "response": "Mock response for query: What is your name?"
    }
    """
    query_embedding = embeddings.embed_query(query.query)
    search_results = vector_store.similarity_search_by_vector(query_embedding, top_k=1)

    if search_results:
        return {"user": username, "response": search_results[0].page_content}
    return {"user": username, "response": "No relevant information found."}
    # return {"response": f"Mock response for query: {query.query}"}

@app.get("/health")
def health_check():
    """
    url - http://127.0.0.1:8000/health
    output:
    {
    "status": "Healthy"
    }
    """
    try:
        qdrant_client.get_collections()
        return {"status": "Healthy"}
    except Exception:
        raise HTTPException(status_code=500, detail="Qdrant connection failed")

# Need run with this command `uvicorn main:app --reload`
