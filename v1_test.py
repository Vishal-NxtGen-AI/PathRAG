import os
from dotenv import load_dotenv
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import nvidia_openai_complete

# Load environment variables from .env file
load_dotenv()

WORKING_DIR = "/Users/1500/Desktop/PathRAG"

api_key=""
os.environ["OPENAI_API_KEY"] = api_key


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=nvidia_openai_complete,  
)

data_file=""
question=""
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














