# build_vector_store.py

import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_pdf_documents(pdf_folder):
    pdf_paths = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    documents = []
    for path in pdf_paths:
        print(f"[PDF] Loading: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents

def load_text_documents(text_folder):
    documents = []
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            path = os.path.join(text_folder, filename)
            print(f"[TXT] Loading: {path}")
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()

            entries = content.split("}\n")  # Split based on end of each JSON block
            for entry in entries:
                if "{" not in entry:
                    continue
                try:
                    text_part = entry.split("{")[0].strip()
                    json_block = "{" + entry.split("{")[1].strip()  # recover full JSON
                    metadata = json.loads(json_block)

                    doc = Document(page_content=text_part, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    print(f"[WARN] Skipping malformed entry: {e}")
    return documents

def main():
    base_dir = os.path.dirname(__file__)
    pdf_folder = os.path.join(base_dir, "pdfs")
    text_folder = os.path.join(base_dir, "text_data")
    persist_directory = os.path.join(base_dir, "chroma_store")

    print("[INFO] Loading PDF documents...")
    pdf_docs = load_pdf_documents(pdf_folder)

    print("[INFO] Loading structured text documents...")
    text_docs = load_text_documents(text_folder)

    all_documents = pdf_docs + text_docs
    print(f"[INFO] Total documents loaded: {len(all_documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()

    print(f"[SUCCESS] Vector store saved to: {persist_directory}")

if __name__ == "__main__":
    main()
