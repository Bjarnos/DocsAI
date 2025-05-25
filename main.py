from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import requests
import os
import time

app = FastAPI()

DISCORD_WEBHOOK_URL = os.getenv("WEBHOOK")

def send_discord_log(message: str):
    print(f"[LOG]: {message}", flush=True)
    if not DISCORD_WEBHOOK_URL:
        print("❌ WEBHOOK not set; cannot send to Discord.", flush=True)
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print(f"❌ Failed to send Discord log: {e}", flush=True)

allowed_origins = [
    "https://docs.bjarnos.dev",
    "https://www.docs.bjarnos.dev",
]

def cors_origin_checker(origin: str) -> bool:
    if origin in allowed_origins:
        return True
    
    if origin.endswith(".docs.bjarnos.dev"):
        return True
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://docs.bjarnos.dev",
        "https://www.docs.bjarnos.dev",
    ],
    allow_origin_regex=r"^https://([a-zA-Z0-9-]+\.)?docs\.bjarnos\.dev$",
    allow_methods=["*"],
    allow_headers=["*"],
)

send_discord_log("🚀 Starting server setup...")

try:
    model_path = "/persistent-models/TinyLlama-1.1b-Chat-v1.0.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        send_discord_log("📥 Model not found, downloading from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            local_dir="/persistent-models"
        )
        send_discord_log("✅ Model downloaded.")
    else:
        send_discord_log("✅ Model found on disk.")

except Exception as e:
    send_discord_log(f"❌ Failed during model check/download: {e}")
    raise

try:
    send_discord_log("🧠 Initializing LlamaCpp model...")
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_batch=8,
        n_gpu_layers=0,
        n_threads=4,
        temperature=0.7,
        stop=["User:", "Assistant:"]
    )
    send_discord_log("✅ LlamaCpp initialized.")
except Exception as e:
    send_discord_log(f"❌ LlamaCpp load failed: {e}")
    raise
try:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = None
    retriever = None
    send_discord_log("✅ Text splitter and embedding model set up.")
except Exception as e:
    send_discord_log(f"❌ Failed to set up text splitter or embedding model: {e}")
    raise

class QueryRequest(BaseModel):
    query: str

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Answer concisely and only the user's question. "
        "Shorter answers are better. "
        "Do NOT include any code snippets, extra formatting, or unrelated information.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    ),
)

@app.on_event("startup")
def load_documents_from_docs_folder():
    send_discord_log("🔁 Running startup document loader...")
    global vector_store, retriever
    all_documents = []

    try:
        for root, _, files in os.walk("docs"):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    loader = TextLoader(path, encoding="utf-8")
                    try:
                        documents = loader.load()
                        all_documents.extend(documents)
                        send_discord_log(f"📄 Loaded {file}")
                    except Exception as e:
                        send_discord_log(f"❌ Failed loading {file}: {e}")
    except Exception as e:
        send_discord_log(f"❌ Unexpected error during docs walk: {e}")
        raise

    if all_documents:
        try:
            texts = text_splitter.split_documents(all_documents)
            vector_store = FAISS.from_documents(texts, embedding_model)
            retriever = vector_store.as_retriever()
            send_discord_log(f"✅ Indexed {len(all_documents)} .md files from docs/")
        except Exception as e:
            send_discord_log(f"❌ Failed to build vector store: {e}")
            raise
    else:
        send_discord_log("⚠️ No markdown documents found in docs/ folder.")

    send_discord_log("✅ Startup complete.")

rate_limits = {}
RATE_LIMIT_MAX = 10
RATE_LIMIT_WINDOW = 3600

def cleanup_old_requests(ip):
    now = time.time()
    if ip in rate_limits:
        rate_limits[ip] = [t for t in rate_limits[ip] if now - t < RATE_LIMIT_WINDOW]
        if not rate_limits[ip]:
            del rate_limits[ip]

@app.post("/ask")
async def ask_question(request: Request, query: QueryRequest):
    client_ip = request.client.host
    cleanup_old_requests(client_ip)
    if client_ip not in rate_limits:
        rate_limits[client_ip] = []
    if len(rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        send_discord_log(f"⚠️ Rate limit exceeded for IP {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded: max 10 requests per hour")

    send_discord_log("📩 Received `/ask` request.")
    if retriever is None:
        send_discord_log("⚠️ Tried asking but no documents indexed.")
        return {"error": "No documents indexed"}
    try:
        qa = create_retrieval_chain(retriever=retriever, llm=llm, prompt=QA_PROMPT)
        result = qa.invoke({"query": query.query})
        answer = str(result)
        send_discord_log(f"📨 Query: {query.query}\n💬 Answer: {answer}")

        rate_limits[client_ip].append(time.time())

        return {"answer": answer}
    except Exception as e:
        send_discord_log(f"❌ Error during QA retrieval: {e}")
        return {"error": str(e)}

@app.get("/")
async def homepage():
    send_discord_log("📡 Received request at `/`")
    return {"success": True}

@app.get("/ping")
async def ping():
    return {"pong": True}
