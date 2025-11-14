# Local LLM RAG Index for Obsidian Vault

This project provides a **fully local Retrieval-Augmented Generation (RAG)** pipeline for querying an Obsidian vault using local models served through **Ollama**.  
All processing (indexing, embeddings, and inference) remains on-device.

---

## Features

- Works entirely offline
- Uses **Ollama** for both LLM + embeddings
- Handles `.md`, `.txt`, and `.pdf` files
- Automatically stores a persistent vector index
- **Only rebuilds the index when files actually change**
- Writes answers back into the vault under `/Generated/`

---

## Requirements

- macOS or Linux
- Python 3.11+
- [Ollama](https://ollama.com/) installed
- ~6–16 GB RAM depending on model choice

### Recommended Ollama Models

| Purpose        | Model        | Notes                         |
| -------------- | ------------ | ----------------------------- |
| **LLM**        | `mistral:7b` | Fast, low RAM, good reasoning |
| **Embeddings** | `all-minilm` | Small + stable for retrieval  |

Pull them:

```sh
ollama pull mistral:7b
ollama pull all-minilm
```


running RAG:

activate the python environment, run `source .venv/bin/activate` in the project 
rootthen run the `rag.py` via `python3 rag.py` in a seperate terminal tab start
up `watch.py` in the same way
