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
    local_dir="models",
    local_dir_use_symlinks=False
)

llm = LlamaCpp(
    model_path=path,
    n_ctx=4096,
    n_batch=512,
    n_gpu_layers=50,
    n_threads=8,
    temperature=0.1
)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None
retriever = None

DISCORD_WEBHOOK_URL = os.getenv("WEBHOOK")

def send_discord_log(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print(f"Failed to send Discord log: {e}", flush=True)

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def load_documents_from_docs_folder():
    global vector_store, retriever
    all_documents = []

    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                loader = TextLoader(path, encoding="utf-8")
                try:
                    documents = loader.load()
                    all_documents.extend(documents)
                    send_discord_log(f"📄 Loaded: {file}")
                except Exception as e:
                    send_discord_log(f"❌ Failed loading {file}: {e}")

    if all_documents:
        texts = text_splitter.split_documents(all_documents)
        vector_store = FAISS.from_documents(texts, embedding_model)
        retriever = vector_store.as_retriever()
        send_discord_log(f"✅ Indexed {len(all_documents)} .md files from docs/")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    send_discord_log("Received request.")
    if retriever is None:
        send_discord_log("⚠️ Tried asking but no documents indexed.")
        return {"error": "No documents indexed"}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    result = qa.invoke({"query": request.query})
    answer = result.get("result", None)

    send_discord_log(f"📨 Query: {request.query}\n💬 Answer: {answer}")
    return {"answer": answer}
