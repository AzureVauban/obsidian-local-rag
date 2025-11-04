import os
import datetime
from pathlib import Path
import hashlib

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser

# Local models via Ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# PDF -> text
import fitz  # PyMuPDF


# --------- CONFIG ---------
VAULT_PATH = (
    "/Users/michaelelder/Documents/Documents/Obsidian Vaults/Personal-Obsidian-Vault"
)
OUTPUT_DIR = os.path.join(VAULT_PATH, "Generated")
PERSIST_DIR = str(Path.home() / "local-llm-rag" / "index_store")

INCLUDE_TOP_LEVEL = [
    "02 - Projects",
    "03 Fall 2025 - TXST",
    "04 - Personal Anthology & C",
]

EXCLUDE_GLOBAL = [
    "00 - Templates",
    "01 - Programming",
    "99 - Meta",
    "Excalidraw",
    "Generated",
]

EXCLUDE_PROJECT_SUBDIRS = [
    "Reactive Wordle",
    "Pet of the Day",
]

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5
STORAGE_DIR = "./storage"

# --------- MODEL CONFIG SWITCH ---------
MODEL_CONFIG = {
    "llm_model": "mistral:7b",
    # Stable embedding model for local Macs
    "embed_model": "all-minilm",
    "embed_truncate": True,
}


# --------- HASHING (Avoid Rebuilding) ----------
def compute_vault_hash() -> str:
    hasher = hashlib.sha256()

    for root, dirs, files in os.walk(VAULT_PATH):
        if any(part in EXCLUDE_GLOBAL for part in Path(root).parts):
            continue
        if not is_allowed_path(root):
            continue
        if is_blocked_project_path(root):
            continue

        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
                hasher.update(str(stat.st_mtime_ns).encode())
                hasher.update(str(stat.st_size).encode())
            except:
                continue
    return hasher.hexdigest()


HASH_FILE = str(Path(PERSIST_DIR) / "vault_hash.txt")


def is_allowed_path(root: str) -> bool:
    base = os.path.basename(root)
    if base in EXCLUDE_GLOBAL:
        return False
    va = os.path.join(VAULT_PATH, "")
    return any(root.startswith(os.path.join(va, inc)) for inc in INCLUDE_TOP_LEVEL)


def is_blocked_project_path(root: str) -> bool:
    if "02 - Projects" not in root:
        return False
    parts = set(Path(root).parts)
    return any(bad in parts or bad in root for bad in EXCLUDE_PROJECT_SUBDIRS)


# --------- PDF Handling ---------
def pdf_to_text(pdf_path: str) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            text_parts = []
            for page in doc:
                try:
                    text_parts.append(page.get_text("text"))
                except:
                    continue
            return "\n".join(text_parts).strip()
    except:
        print(f"[WARN] Skipping unreadable PDF: {pdf_path}")
        return ""


# --------- Document Collection ---------
def collect_documents() -> list[Document]:
    from tqdm import tqdm

    docs: list[Document] = []
    all_files = []

    for root, dirs, files in os.walk(VAULT_PATH):
        if any(part in EXCLUDE_GLOBAL for part in Path(root).parts):
            continue
        if not is_allowed_path(root):
            continue
        if is_blocked_project_path(root):
            continue
        for fname in files:
            all_files.append(os.path.join(root, fname))

    print(f"\nIndexing {len(all_files)} files...\n")

    for fpath in tqdm(all_files, desc="Collecting Documents", unit="files"):
        ext = os.path.splitext(fpath)[1].lower()

        if ext in (".md", ".txt"):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    rel = os.path.relpath(fpath, VAULT_PATH)
                    docs.append(Document(text=text, metadata={"source": rel}))
            except:
                continue

    return docs


# --------- Index Build / Load ---------
def build_or_load_index(embed_model) -> VectorStoreIndex:
    from tqdm import tqdm

    current_hash = compute_vault_hash()

    # ✅ If unchanged, load existing index
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, "r") as f:
                if f.read().strip() == current_hash:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=PERSIST_DIR
                    )
                    index = load_index_from_storage(storage_context)
                    print("[+] No vault changes detected. Using existing index.\n")
                    return index
        except:
            pass

    print("[!] Vault changed — rebuilding index.\n")

    documents = collect_documents()
    parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=32)
    nodes = []

    print("\n[+] Chunking documents...\n")
    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        try:
            nodes.extend(parser.get_nodes_from_documents([doc]))
        except:
            continue

    print("\n[+] Embedding + Constructing Vector Index...\n")
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(current_hash)

    print("[+] Index updated and hash stored.\n")
    return index


# --------- Logging Output to Vault ---------
def write_answer(query: str, answer: str) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(OUTPUT_DIR) / f"RAG_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Q: {query}\n\n{answer}\n")
    print(f"\nSaved to: {out_path}\n")


# --------- MAIN LOOP ---------
def main() -> None:
    llm = Ollama(model=MODEL_CONFIG["llm_model"])
    embed_model = OllamaEmbedding(
        model_name=MODEL_CONFIG["embed_model"],
        truncate=MODEL_CONFIG["embed_truncate"],
    )

    index = build_or_load_index(embed_model=embed_model)
    query_engine = index.as_query_engine(similarity_top_k=TOP_K, llm=llm)

    print("\nRAG is ready. Type a question below. (exit/quit to stop)\n")
    while True:
        q = input("Ask: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        resp = query_engine.query(q)
        ans = str(resp)
        print("\n" + ans + "\n")
        write_answer(q, ans)


if __name__ == "__main__":
    main()
