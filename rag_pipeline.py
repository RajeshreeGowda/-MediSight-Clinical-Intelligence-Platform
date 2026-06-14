import os
import sys
import time
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Step 1: Check discharge_notes ─────────────────────────────
print("Step 1: Checking discharge_notes folder...")
if not os.path.exists("discharge_notes") or \
   len([f for f in os.listdir("discharge_notes") if f.endswith(".txt")]) == 0:
    print("ERROR: No .txt files in discharge_notes/")
    print("Run first: python create_notes.py")
    sys.exit(1)

txt_files = [f for f in os.listdir("discharge_notes") if f.endswith(".txt")]
print("  Found", len(txt_files), "text files")

# ── Step 2: Load + chunk (with encoding fix) ───────────────────
print("Step 2: Loading and chunking documents...")
docs, ids, metas = [], [], []

for fname in txt_files:
    path = os.path.join("discharge_notes", fname)

    # Try multiple encodings — fixes UnicodeDecodeError
    content = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            content = open(path, encoding=enc).read().strip()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        # Last resort — ignore bad bytes
        content = open(path, encoding="utf-8", errors="ignore").read().strip()
        print("  Warning: used 'ignore' encoding for", fname)

    # Remove any non-ASCII characters to be safe
    content = content.encode("ascii", errors="ignore").decode("ascii")

    if not content:
        print("  Skipping empty file:", fname)
        continue

    # Split into ~280 char chunks
    words  = content.split()
    chunks = []
    chunk  = []
    for w in words:
        chunk.append(w)
        if len(" ".join(chunk)) >= 280:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))

    for i, ch in enumerate(chunks):
        docs.append(ch)
        ids.append(fname.replace(".txt", "") + "_chunk_" + str(i))
        metas.append({"source": fname})

print("  Created", len(docs), "chunks from", len(txt_files), "files")

# ── Step 3: Load embedding model ──────────────────────────────
print("Step 3: Loading embedding model (first time ~2 min)...")
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
print("  Embedding model ready")

# ── Step 4: Delete old chroma_db (Windows-safe) ───────────────
print("Step 4: Removing old chroma_db...")
if os.path.exists("./chroma_db"):
    for attempt in range(5):
        try:
            for root, dirs, flist in os.walk("./chroma_db"):
                for f in flist:
                    try:
                        os.chmod(os.path.join(root, f), 0o777)
                    except Exception:
                        pass
            shutil.rmtree("./chroma_db", ignore_errors=True)
            if not os.path.exists("./chroma_db"):
                print("  Removed old chroma_db")
                break
        except Exception:
            if attempt < 4:
                print("  Retrying (" + str(attempt+1) + "/5)...")
                time.sleep(1)
            else:
                print("  Could not delete automatically.")
                print("  Please stop Streamlit (Ctrl+C) and manually delete the chroma_db folder,")
                print("  then run this script again.")
                sys.exit(1)

# ── Step 5: Build ChromaDB ─────────────────────────────────────
print("Step 5: Building vector database...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("discharge_notes")
except Exception:
    pass

collection = client.create_collection(
    name="discharge_notes",
    embedding_function=ef
)

# Add in batches of 50
batch = 50
for i in range(0, len(docs), batch):
    collection.add(
        documents=docs[i:i+batch],
        ids=ids[i:i+batch],
        metadatas=metas[i:i+batch]
    )
    print("  Indexed", min(i+batch, len(docs)), "/", len(docs), "chunks")

print()
print("=" * 45)
print("  RAG pipeline complete!")
print("  Chunks indexed:", collection.count())
print("  Run: streamlit run app.py")
print("=" * 45)