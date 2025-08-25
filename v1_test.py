import os
from functools import partial
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import nvidia_nims_complete, nvidia_nims_embedding

WORKING_DIR = "/Users/1500/Developer/PathRAG"

# NVIDIA NIMs Configuration
# Replace these with your actual NVIDIA NIMs endpoints
NVIDIA_NIMS_LLM_ENDPOINT = "http://your-nims-host:8000/v1"  # Replace with your LLM endpoint
NVIDIA_NIMS_EMBEDDING_ENDPOINT = "http://your-nims-host:8001/v1"  # Replace with your embedding endpoint
NVIDIA_NIMS_API_KEY = ""  # Replace with your API key if needed

# LLM model name served by your NIMs deployment
NVIDIA_NIMS_LLM_MODEL = "llama-3.1-8b-instruct"  # Replace with your actual model name
# Embedding model name served by your NIMs deployment  
NVIDIA_NIMS_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"  # Replace with your actual embedding model name


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Create custom wrapper functions for your NVIDIA NIMs endpoints
def custom_nvidia_nims_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    """Wrapper function for NVIDIA NIMs LLM with your endpoints"""
    return nvidia_nims_complete(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=NVIDIA_NIMS_LLM_ENDPOINT,
        model_name=NVIDIA_NIMS_LLM_MODEL,
        api_key=NVIDIA_NIMS_API_KEY,
        **kwargs
    )

def custom_nvidia_nims_embedding(texts, **kwargs):
    """Wrapper function for NVIDIA NIMs embedding with your endpoints"""
    return nvidia_nims_embedding(
        texts=texts,
        base_url=NVIDIA_NIMS_EMBEDDING_ENDPOINT,
        model=NVIDIA_NIMS_EMBEDDING_MODEL,
        api_key=NVIDIA_NIMS_API_KEY,
        **kwargs
    )

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=custom_nvidia_nims_llm,
    embedding_func=custom_nvidia_nims_embedding,
)

data_file=""
question=""
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














