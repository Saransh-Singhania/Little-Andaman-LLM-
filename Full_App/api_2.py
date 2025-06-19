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
    top_k: int = 1

# Globals
vector_store = None

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
        # Retrieve relevant documents
        docs = await asyncio.to_thread(
            vector_store.similarity_search, request.prompt, request.top_k
        )

        if not docs:
            return {
                "response": "Sorry, I cannot answer that based on the available documents.",
                "context_sources": []
            }

        context = "\n\n".join([doc.page_content for doc in docs])
        if len(context) > 3000:
            context = context[:3000]

        prompt = textwrap.dedent(f"""
            You are a helpful assistant who answers questions **only based on the following documents** related to Little Andaman Island and its development projects.

            **If the answer is not present in the provided context, reply with: "Sorry, I cannot answer that based on the available documents."**
                                 
            **Do not make up answers or provide information not found in the context.**
                                 
            **Don't show your think process or reasoning. Just provide the final answer.**

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
                    "stream": False
                },
                timeout=60.0
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ollama model error.")

        result = response.json()
        response_text = result.get("response", "").strip()

        return {
            "response": response_text,
            "context_sources": [
                {"source": doc.metadata.get("source", "unknown"), "text": doc.page_content}
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
