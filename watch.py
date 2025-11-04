import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

VAULT_PATH = (
    "/Users/michaelelder/Documents/Documents/Obsidian Vaults/Personal-Obsidian-Vault"
)
REBUILD_DELAY = 5  # seconds after last save to rebuild

pending = False
lock = threading.Lock()


def trigger_rebuild():
    global pending
    with lock:
        pending = True


class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        trigger_rebuild()

    def on_created(self, event):
        trigger_rebuild()

    def on_deleted(self, event):
        trigger_rebuild()


def rebuild_loop():
    global pending
    while True:
        time.sleep(1)
        with lock:
            if pending:
                pending = False
                time.sleep(REBUILD_DELAY)

                print("\n[Watcher] Changes detected → Rebuilding index...\n")
                subprocess.run(["python3", "rag.py"])


if __name__ == "__main__":
    print("[Watcher] Watching for changes...")
    observer = Observer()
    observer.schedule(Handler(), VAULT_PATH, recursive=True)
    observer.start()

    try:
        rebuild_loop()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
