#!/opt/et_replica1/venv/bin/python3
import json, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

ROOT = Path("/opt/et_replica1/data/corpus")
IDX_DIR = Path("/opt/et_replica1/index"); IDX_DIR.mkdir(parents=True, exist_ok=True)
IDX_BIN = IDX_DIR/"etomega.faiss"
META_JSON = IDX_DIR/"etomega_meta.json"

def read_txt(p: Path) -> str:
    b = p.read_bytes()
    try:
        import chardet; enc = chardet.detect(b).get("encoding") or "utf-8"
    except Exception:
        enc = "utf-8"
    return b.decode(enc, errors="ignore")

def read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in r.pages)
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(p)) or ""
        except Exception:
            print(f"[WARN] {p}: pdf ilegível", file=sys.stderr)
            return ""

def read_docx(p: Path) -> str:
    try:
        import docx2txt
        return docx2txt.process(str(p)) or ""
    except Exception:
        print(f"[WARN] {p}: docx inválido", file=sys.stderr)
        return ""

def chunk_text(t: str, size=800, overlap=80):
    out = []
    i = 0; n = len(t)
    while i < n:
        out.append(t[i:i+size])
        i += size - overlap
    return [c for c in out if c.strip()]

def read_any(p: Path) -> str:
    s = p.suffix.lower()
    if s in {".txt",".md",".log"}: return read_txt(p)
    if s == ".pdf": return read_pdf(p)
    if s == ".docx": return read_docx(p)
    return ""

files = sorted([p for p in ROOT.rglob("*") if p.is_file()])
texts = []
for p in tqdm(files, desc="lendo"):
    try:
        t = read_any(p)
        if t: texts.extend(chunk_text(t))
    except Exception as e:
        print(f"[WARN] {p}: {e}", file=sys.stderr)

if not texts:
    print("[erro] nenhum texto coletado", file=sys.stderr); sys.exit(2)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

faiss.write_index(index, str(IDX_BIN))
json.dump({"chunks": texts, "meta": {"emb":"all-MiniLM-L6-v2","metric":"ip","n":len(texts)}}, 
          open(META_JSON, "w", encoding="utf-8"), ensure_ascii=False)

print(f"[ingest] chunks: {len(texts)}")
print(f"[ok] índice salvo: {IDX_BIN} meta: {META_JSON}")
