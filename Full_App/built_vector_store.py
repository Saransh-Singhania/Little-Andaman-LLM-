# build_vector_store.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load all PDFs from folder
pdf_folder = os.path.join(os.path.dirname(__file__), "pdfs")
pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

documents = []
for path in pdf_paths:
    print(f"Loading: {path}")
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

print(f"Loaded {len(documents)} documents.")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# Embedding model (local HuggingFace)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build Chroma vectorstore
persist_directory = os.path.join(os.path.dirname(__file__), "chroma_store")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)
vectorstore.persist()
print(f"Chroma vector store saved to: {persist_directory}")
