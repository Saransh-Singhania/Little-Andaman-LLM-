# https://ollama.com/download --> install it and it will run in the background to facilitate operating llm,s locally.
# In terminal (powershell) run "ollama run qwen3:0.6b" --> this ollama run command let's us download our model of choice form the list of open source models available freely on ollama 
# pip install fastapi uvicorn pydantic httpx langchain langchain-community chromadb huggingface-hub sentence-transformers  --> Run in terminal. this is a  list of required python packages to be installed in vscode before executing any script.
# Delete existing pdf and all files/folders in "chroma_store" and add your own pdfs to the pdfs folder.
# Run the script to build the vector store. The vector files will be automatically written to the chroma_store sirectory
# Execute uvicorn api_2:app --reload in terminal to run the api sciprt
# Open http://localhost:8000/docs to test the API


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import asyncio
import httpx
import textwrap

# ---- FastAPI setup ----
app = FastAPI(title="Offline RAG API (Chroma + Qwen via Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Generation and retrieval parameters (backend-controlled). These control how the model can respond.
TEMPERATURE = 0.2
TOP_P = 0.9
REPEAT_PENALTY = 1.1
MAX_TOKENS = 250
TOP_K = 2 # <--- Fixed top_k value for similarity search

# Globals
vector_store = None
MAX_CONTEXT_LENGTH = 1500  # Maximum context length for the model 

@app.on_event("startup")
async def load_components():
    global vector_store

    print("[Startup] Loading embeddings & vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory="chroma_store",
        embedding_function=embeddings
    )
    print("[Startup] Vector store ready.")

@app.post("/generate", response_model=Dict[str, object])
async def generate_response(request: PromptRequest):
    try:
        # Retrieve relevant documents with scores
        docs = await asyncio.to_thread(
            vector_store.similarity_search_with_score, request.prompt, TOP_K
        )

        if not docs:
            return {
                "response": "Sorry, I cannot answer that based on the available documents.",
                "context_sources": []
            }

        # Concatenate context
        context = "\n\n".join([doc[0].page_content for doc in docs])
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]

        prompt = textwrap.dedent(f"""
            You are an expert on Little Andaman Island.
                                 
            Always assume the user is asking about Little Andaman Island even if they don't explicitly mention it.

            Use ALL of the following information to answer comprehensively.

            - Be clear and detailed.
            - If multiple questions are asked, answer all of them.
            - If the answer is not in the context, say: "Sorry, I cannot answer that based on the available documents."
            - DO NOT make up information.
            - DO NOT show your reasoning.

            [Context Starts]
            {context}
            [Context Ends]

            Question: {request.prompt}

            Answer:
        """)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "repeat_penalty": REPEAT_PENALTY,
                    "max_tokens": MAX_TOKENS
                },
                timeout=120.0
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ollama model error.")

        result = response.json()
        response_text = result.get("response", "").strip()

        return {
            "response": response_text,
            "context_sources": [
                {
                    "source": doc[0].metadata.get("source", "unknown"),
                    "text": doc[0].page_content,
                    "score": doc[1]
                }
                for doc in docs
            ]
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy" if vector_store else "unhealthy"}
