#Flan version of code

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import asyncio
import textwrap

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---- FastAPI setup ----
app = FastAPI(title="Offline RAG API (Chroma + Flan-T5)")

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
    top_k: int = 3

# Globals
vector_store = None
generation_pipeline = None

@app.on_event("startup")
async def load_components():
    global vector_store, generation_pipeline

    print("[Startup] Loading embeddings & vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory="chroma_store",
        embedding_function=embeddings
    )

    print("[Startup] Loading FLAN-T5 model...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    generation_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )

    print("[Startup] RAG system ready.")

@app.post("/generate", response_model=Dict[str, object])
async def generate_response(request: PromptRequest):
    try:
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
            context = context[:3000]  # truncate context if too long

        # Formulate FLAN prompt
        prompt = textwrap.dedent(f"""
            Answer the question based only on the context below.
            If the answer is not present, say "Sorry, I cannot answer that based on the available documents."

            Context:
            {context}

            Question: {request.prompt}
            Answer:
        """)

        outputs = await asyncio.to_thread(
            generation_pipeline,
            prompt,
            max_new_tokens=256,
            do_sample=False
        )
        response_text = outputs[0]['generated_text'].strip()

        return {
            "response": response_text,
            "context_sources": [
                {"source": doc.metadata.get("source", "unknown"), "text": doc.page_content}
                for doc in docs
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy" if vector_store else "unhealthy"}
