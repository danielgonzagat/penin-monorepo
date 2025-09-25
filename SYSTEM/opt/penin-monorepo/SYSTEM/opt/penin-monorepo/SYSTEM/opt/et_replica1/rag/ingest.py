#!/usr/bin/env python3
import sys, os, json, re, pathlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np, faiss
from pypdf import PdfReader
import docx2txt

def read_file(p: pathlib.Path):
    s = p.suffix.lower()
    try:
        if s=='.pdf':
            r=PdfReader(str(p))
            return "\n".join([(pg.extract_text() or "") for pg in r.pages])
        if s in ('.txt','.md','.markdown'):
            return p.read_text(errors='ignore')
        if s=='.docx':
            return docx2txt.process(str(p)) or ""
    except Exception:
        return ""
    return ""

def clean(t: str):
    t = t.replace('\x00','')
    t = re.sub(r'\s+',' ', t)
    return t.strip()

def chunk(t, max_len=900, overlap=150):
    out=[]; i=0; n=len(t)
    while i<n:
        j=min(n, i+max_len)
        out.append(t[i:j])
        if j==n: break
        i = max(0, j-overlap)
    return out

def files(root):
    for p in pathlib.Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.pdf','.txt','.md','.markdown','.docx'):
            yield p

def main():
    corpus = sys.argv[1] if len(sys.argv)>1 else '/opt/et_replica1/data/corpus'
    outdir = '/opt/et_replica1/data/index'
    os.makedirs(outdir, exist_ok=True)
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    dim   = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    metas=[]; total=0
    docs=list(files(corpus))
    for p in tqdm(docs, desc="lendo"):
        raw=read_file(p)
        if not raw: continue
        txt=clean(raw)
        if not txt: continue
        ch=chunk(txt, 900, 150)
        em=model.encode(ch, normalize_embeddings=True, show_progress_bar=False).astype('float32')
        index.add(em)
        for i,c in enumerate(ch):
            metas.append({'source': str(p), 'i': i, 'text': c[:1000]})
        total += len(ch)
    faiss.write_index(index, f'{outdir}/index.faiss')
    with open(f'{outdir}/meta.jsonl','w') as f:
        for m in metas: f.write(json.dumps(m, ensure_ascii=False)+'\n')
    with open(f'{outdir}/model.txt','w') as f:
        f.write(model_name)
    print(f'ok: {total} chunks; dim={dim}; saved in {outdir}')
if __name__=='__main__':
    main()
