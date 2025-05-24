from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import requests
import os

app = FastAPI()

path = hf_hub_download(
    repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3.Q5_K_M.gguf",
    local_dir="models"
)

print(path, flush=True)
