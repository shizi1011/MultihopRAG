import os
import io
from llm import Loadllm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import chainlit as cl

DATA_PATH = 'data'
DB_PATH = 'vectorstore'

# def create_vector_db(docs, embeddings):
#     db = FAISS.from_documents(docs, embeddings)
#     # db.save_local(DB_PATH)
#     return db

async def create_vector_db(docs, embeddings):
    db = await cl.make_async(FAISS.from_documents)(docs, embeddings)
    # db.save_local(DB_PATH)
    return db
def file_ingestor(file_path,chunk_size=1000,chunk_overlap=100):
    loader = PyMuPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs
