#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, logging
from pathlib import Path
from typing import List, Dict, Tuple
from pypdf import PdfReader
import faiss, numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken

logging.basicConfig(level=logging.INFO, format="[ETKB] %(levelname)s: %(message)s")

CORPUS_DIR = Path(os.environ.get("ETKB_CORPUS","/opt/et_fusion/data/corpus"))
INDEX_DIR  = Path(os.environ.get("ETKB_INDEX","/opt/et_fusion/data/index"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("ETKB_EMB","BAAI/bge-m3")
enc = tiktoken.get_encoding("cl100k_base")

SKIP_NAMES = {".ds_store"}
SKIP_SUFFIXES = {".zip", ".docx", ".xlsx", ".pptx", ".csv~", ".tmp"}
SKIP_DIR_HINTS = {"archives"}  # evita /05_Appendix/Archives etc.

def _read_text(path: Path) -> str:
    if path.name.lower() in SKIP_NAMES: return ""
    if any(hint in str(path).lower() for hint in SKIP_DIR_HINTS): return ""
    if path.suffix.lower() in SKIP_SUFFIXES: return ""

    suf = path.suffix.lower()
    try:
        if suf == ".pdf":
            txt=[]; r=PdfReader(str(path))
            for p in r.pages:
                try: txt.append(p.extract_text() or "")
                except Exception as e:
                    logging.warning(f"Falha ao extrair página em {path}: {e}")
                    txt.append("")
            return "\n".join(txt)
        elif suf in [".txt",".md",".markdown"]:
            return path.read_text(errors="ignore", encoding="utf-8")
        else:
            return ""
    except Exception as e:
        logging.warning(f"Arquivo problemático (pulado): {path} | {e}")
        return ""

def _chunk(text:str, max_tokens=400, overlap=80)->List[str]:
    toks=enc.encode(text)
    chunks=[]
    i=0
    while i < len(toks):
        j = min(i+max_tokens, len(toks))
        chunk = enc.decode(toks[i:j])
        chunks.append(chunk)
        i = j - overlap
        if i < 0: i = 0
        if j==len(toks): break
    return [c.strip() for c in chunks if c.strip()]

def build_index()->Tuple[faiss.IndexFlatIP, List[Dict]]:
    logging.info(f"CORPUS_DIR: {CORPUS_DIR}")
    model = SentenceTransformer(MODEL_NAME)
    metas=[]; vecs=[]

    files = sorted(CORPUS_DIR.rglob("*"))
    if not files:
        raise SystemExit(f"Nenhum arquivo em {CORPUS_DIR}")

    for fp in files:
        if fp.is_dir(): continue
        name=fp.name.lower()
        if name.startswith("._"): continue  # artefatos do macOS
        text = _read_text(fp)
        if not text.strip(): continue
        chs = _chunk(text)
        if not chs: continue
        emb = model.encode(chs, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
        for i,(c,e) in enumerate(zip(chs,emb)):
            metas.append({"file": str(fp), "chunk": i, "text": c})
            vecs.append(e)

    if not vecs:
        raise SystemExit("Nenhum texto útil após filtragem. Verifique formatos e PDFs.")

    xb = np.stack(vecs).astype("float32")
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.write_index(index, str(INDEX_DIR/"etkb.index"))
    (INDEX_DIR/"metas.json").write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")
    logging.info(f"OK: índice criado em {INDEX_DIR}")
    return index, metas

def load_index():
    index = faiss.read_index(str(INDEX_DIR/"etkb.index"))
    metas = json.loads((INDEX_DIR/"metas.json").read_text(encoding="utf-8"))
    return index, metas, SentenceTransformer(MODEL_NAME)

def search(query:str, k=8)->Dict:
    index, metas, model = load_index()
    qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
    D,I = index.search(np.array([qv]), k)
    hits=[]
    for rank, idx in enumerate(I[0].tolist()):
        m = metas[idx]
        hits.append({"rank":rank+1,"score":float(D[0][rank]),"file":m["file"],"chunk":m["chunk"],"text":m["text"]})
    context = "\n\n".join([f"[{h['rank']}] {h['file']}#chunk{h['chunk']}\n{h['text']}" for h in hits])
    return {"context": context, "hits": hits}

if __name__=="__main__":
    build_index()
