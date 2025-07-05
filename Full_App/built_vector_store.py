

import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Directories
base_dir = os.path.dirname(__file__)
pdf_folder = os.path.join(base_dir, "pdfs")
text_folder = os.path.join(base_dir, "text_data")
persist_directory = os.path.join(base_dir, "chroma_store")

# Load PDFs
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        print(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

# Load structured text files
for file in os.listdir(text_folder):
    if file.endswith(".txt"):
        path = os.path.join(text_folder, file)
        print(f"Processing structured text file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        buffer, metadata = [], {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    metadata = json.loads(line)
                    doc = Document(page_content="\n".join(buffer).strip(), metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    print(f"Failed to parse metadata in {path}: {e}")
                buffer = []
            else:
                buffer.append(line)

print(f"Total loaded documents: {len(documents)}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Embedding + Vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=persist_directory)
vectorstore.persist()
print(f"[✔] Chroma vector store saved at: {persist_directory}")
