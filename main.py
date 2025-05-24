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

send_discord_log("🚀 Starting server setup...")

try:
    model_path = "/persistent-models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        send_discord_log("📥 Model not found, downloading from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
            filename="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
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
        n_batch=512,
        n_gpu_layers=50,
        n_threads=4,
        temperature=0.1
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

def truncate_text(text, max_chars=500):
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period != -1:
        return truncated[:last_period+1]
    return truncated

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer concisely and only the user's query. "
    "Do not add unrelated code snippets or documentation unless asked."
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

@app.get("/")
async def homepage():
    send_discord_log("📡 Received request at `/`")
    return {"success": True}

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    send_discord_log("📩 Received `/ask` request.")
    if retriever is None:
        send_discord_log("⚠️ Tried asking but no documents indexed.")
        return {"error": "No documents indexed"}

    prompt = SYSTEM_PROMPT + "\nUser: " + request.query + "\nAssistant:"
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )
        result = qa.invoke({"query": prompt})
        answer = result.get("result", None)
        if answer:
            answer = truncate_text(answer, max_chars=500)
        send_discord_log(f"📨 Query: {request.query}\n💬 Answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        send_discord_log(f"❌ Error during QA retrieval: {e}")
        return {"error": str(e)}
