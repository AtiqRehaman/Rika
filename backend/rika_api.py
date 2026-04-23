from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========================
# Initialize FastAPI
# ========================
app = FastAPI()

# ========================
# Load LLM
# ========================
llm = Llama(
    model_path="/content/based-7B-Q4_K_M.gguf",  # change if needed
    n_ctx=2048,
    n_gpu_layers=35,
    verbose=False
)

# ========================
# Minimal RAG Corpus
# ========================
# docs = [
#     "torch.nn.Module is the base class for all neural network modules in PyTorch.",
#     "torch.optim.Adam implements the Adam optimization algorithm.",
#     "torch.utils.data.DataLoader loads data in batches and supports shuffling.",
#     "tf.keras.Model is the base class for building Keras models.",
#     "tf.data.Dataset is used for TensorFlow input pipelines."
# ]

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# doc_embeddings = embedder.encode(docs, convert_to_numpy=True)

# def retrieve(query, top_k=2):
#     query_embedding = embedder.encode([query], convert_to_numpy=True)
#     similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [docs[i] for i in top_indices]

# ========================
# System Prompt
# ========================
RIKA_SYSTEM = """
You are Rika.
You are a precise coding assistant.
You specialize in PyTorch and TensorFlow.
Be concise and accurate.
"""

# ========================
# Memory Store
# ========================
sessions = {}

# ========================
# Request Model
# ========================
class ChatRequest(BaseModel):
    session_id: str
    message: str

# ========================
# Build Prompt
# ========================
def build_prompt(session_id, user_message):
    history = sessions.get(session_id, "")
    # retrieved_docs = retrieve(user_message)
    # context = "\n".join(retrieved_docs)

    prompt = f"""
    ### System:
    {RIKA_SYSTEM}

    ### Conversation:
    {history}
    User: {user_message}
    Assistant:
    """
    return prompt

# ========================
# Chat Endpoint
# ========================
@app.post("/chat")
def chat(req: ChatRequest):
    prompt = build_prompt(req.session_id, req.message)

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1
    )

    response_text = output["choices"][0]["text"].strip()

    # Save memory
    sessions[req.session_id] = sessions.get(req.session_id, "") + \
        f"\nUser: {req.message}\nAssistant: {response_text}"

    return {"response": response_text}
