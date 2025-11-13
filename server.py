from fastapi import FastAPI, Body
from llama_index.core import load_index_from_storage, StorageContext
from rag import MODEL_CONFIG, STORAGE_DIR
from llama_index.llms.ollama import Ollama

app = FastAPI()

llm = Ollama(model=MODEL_CONFIG["llm_model"])
storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
index = load_index_from_storage(storage_context, llm=llm)
query_engine = index.as_query_engine(llm=llm)

@app.post("/query")
def query(q: str = Body(..., embed=True)):
    resp = query_engine.query(q)
    return {"answer": str(resp)}
