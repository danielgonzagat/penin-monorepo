#!/opt/et_ultimate/venv/bin/python3
import os, re, json, time, uuid, math, logging, subprocess, difflib
from pathlib import Path
from datetime import datetime
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def _json_sanitize(o):
    if isinstance(o, dict):
        return {str(k): _json_sanitize(v) for k,v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_sanitize(x) for x in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return _json_sanitize(o.tolist())
    return o


# --------- ENV ----------
ET_URL   = os.getenv("ET_URL", "http://127.0.0.1:8080/v1")
ET_KEY   = os.getenv("ET_API_KEY", "DANIEL")
ET_MODEL = os.getenv("ET_MODEL", "").strip()
INDEX_P  = Path(os.getenv("ET_INDEX", "/opt/et_ultimate/index/etomega.faiss"))
META_P   = Path(os.getenv("ET_META",  "/opt/et_ultimate/index/etomega_meta.json"))
PROMPT_P = Path(os.getenv("ET_PROMPT", "/opt/et_ultimate/prompts/et_omega.txt"))
REPO_DIR = Path(os.getenv("ET_REPO",   "/opt/et_ultimate/equation"))
LOGFILE  = os.getenv("ET_LOG", "/var/log/et_ultimate/evolver.log")

EI_MIN         = float(os.getenv("EI_MIN", "0.05"))
ENTROPY_MIN    = float(os.getenv("ENTROPY_MIN", "3.0"))
DIV_JSD_MAX    = float(os.getenv("DIV_JSD_MAX", "0.15"))
DRIFT_MAX      = float(os.getenv("DRIFT_MAX", "0.35"))
COST_TOK_MAX   = int(os.getenv("COST_TOK_MAX", "2048"))
VAR_MIN        = float(os.getenv("VAR_MIN", "0.15"))

SLEEP_SECONDS  = int(os.getenv("SLEEP_SECONDS", "60"))
TOPK_CTX       = int(os.getenv("TOPK_CTX", "6"))
CTX_CHARS      = int(os.getenv("CTX_CHARS", "8000"))

STATE_FILE = REPO_DIR / "ET_state.md"

# --------- LOG ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGFILE), logging.StreamHandler()]
)
log = logging.getLogger("evolver")

# --------- HELPERS ----------
def ensure_git_repo():
    if not (REPO_DIR/".git").exists():
        REPO_DIR.mkdir(parents=True, exist_ok=True)
        run(["git","init","-q"], cwd=REPO_DIR)
        run(["git","config","user.name","ET Evolver"], cwd=REPO_DIR)
        run(["git","config","user.email","et@local"], cwd=REPO_DIR)
        if not STATE_FILE.exists():
            STATE_FILE.write_text("# ET_state.md (baseline)\n\n", encoding="utf-8")
            run(["git","add","ET_state.md"], cwd=REPO_DIR)
            run(["git","commit","-q","-m","baseline: ET_state.md"], cwd=REPO_DIR)

def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)

def get_model_id():
    if ET_MODEL:
        return ET_MODEL
    try:
        r = requests.get(f"{ET_URL}/models", timeout=10).json()
        return r["data"][0]["id"]
    except Exception as e:
        log.error(f"falha /models: {e}")
        return None

def approx_tokens(text:str)->int:
    # aproximação “4 chars ~ 1 token”
    return max(1, len(text)//4)

def entropy_bits_per_char(text:str)->float:
    if not text: return 0.0
    from collections import Counter
    c = Counter(text)
    total = sum(c.values())
    H = -sum((n/total)*math.log2(n/total) for n in c.values())
    return H

def js_divergence(a:str,b:str)->float:
    # histograma de caracteres (rápido e estável)
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    chars = set(ca)|set(cb)
    pa = np.array([ca.get(ch,0) for ch in chars], dtype=float)
    pb = np.array([cb.get(ch,0) for ch in chars], dtype=float)
    if pa.sum()==0 or pb.sum()==0: return 0.0
    pa/=pa.sum(); pb/=pb.sum()
    m = 0.5*(pa+pb)
    def kl(p,q): 
        nz = p>0
        return np.sum(p[nz]*np.log2(p[nz]/q[nz]))
    return 0.5*kl(pa,m)+0.5*kl(pb,m)

def vocab_diversity(text:str)->float:
    import re
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", text.lower())
    if not toks: return 0.0
    return len(set(toks))/len(toks)

def drift_ratio(old:str,new:str)->float:
    sm = difflib.SequenceMatcher(None, old, new)
    same = sum(tr.size for tr in sm.get_matching_blocks())
    total = max(len(old), len(new), 1)
    return 1.0 - (same/total)

def read_prompt():
    if PROMPT_P.exists():
        return PROMPT_P.read_text(encoding="utf-8").strip()
    return ("Você é a ET Ultimate do servidor do Daniel. "
            "Responda somente com o bloco <EQUATION>...</EQUATION> "
            "e um <EISCORE>0.xx</EISCORE> em nova linha.")

# --------- RAG ----------
class RAG:
    def __init__(self, index_path:Path, meta_path:Path):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.chunks = meta["chunks"]
        self.info   = meta["meta"]
    def topk(self, query:str, k:int)->list[str]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        D,I = self.index.search(q, k)
        out=[]
        for idx in I[0]:
            if 0 <= idx < len(self.chunks):
                out.append(self.chunks[idx])
        return out

# --------- MODEL CALL ----------
def chat_completion(model_id, messages, max_tokens=512, temperature=0.2):
    headers = {"Content-Type":"application/json"}
    if ET_KEY: headers["Authorization"] = f"Bearer {ET_KEY}"
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    r = requests.post(f"{ET_URL}/chat/completions", headers=headers, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

EQU_RE = re.compile(r"<EQUATION>(.*?)</EQUATION>", re.S|re.I)
EI_RE  = re.compile(r"<EISCORE>\s*([0-9]*\.?[0-9]+)\s*</EISCORE>", re.I)

def propose_new_state(model_id, rag:RAG, old_text:str)->tuple[str,float,str]:
    prompt_sys = read_prompt()
    # consulta contextual
    q = "ETΩ: proponha micro-atualização segura que maximize Expected Improvement mantendo guardrails."
    ctx = "\n\n".join(rag.topk(q, TOPK_CTX))[:CTX_CHARS]
    # resumo curto do estado atual para o modelo
    old_preview = old_text[:2000]

    messages = [
        {"role":"system","content": prompt_sys + (
            "\n\nFORMATO DE SAÍDA OBRIGATÓRIO:\n"
            "<EQUATION>CONTEUDO_COMPLETO_ATUALIZADO</EQUATION>\n"
            "<EISCORE>0.xx</EISCORE>\n"
            "Não escreva nada fora desses blocos.")},
        {"role":"user","content":
         f"Contexto (amostras RAG):\n{ctx}\n\n"
         f"Estado atual (preview):\n{old_preview}\n\n"
         "Tarefa: proponha UMA melhoria pequena, conservadora e incremental na Equação de Turing "
         "(ETΩ), mantendo entropia mínima, divergência limitada, drift controlado, orçamento de custo "
         "e variância mínima do currículo. Evite mudanças amplas. Responda no formato exigido."
        }
    ]
    out = chat_completion(model_id, messages, max_tokens=800, temperature=0.2)
    eqm = EQU_RE.search(out or "")
    eim = EI_RE.search(out or "")
    if not eqm:
        return "", -1.0, out or ""
    new_text = eqm.group(1).strip()
    try:
        ei = float(eim.group(1)) if eim else -1.0
    except:
        ei = -1.0
    return new_text, ei, out

def guardrails_ok(old_text:str, new_text:str, ei:float)->tuple[bool,dict]:
    ent = entropy_bits_per_char(new_text)
    jsd = js_divergence(old_text, new_text)
    drift = drift_ratio(old_text, new_text)
    var = vocab_diversity(new_text)
    cost = approx_tokens(new_text)

    checks = {
        "ei_ok": ei >= EI_MIN,
        "entropy_ok": ent >= ENTROPY_MIN,
        "div_ok": jsd <= DIV_JSD_MAX,
        "drift_ok": drift <= DRIFT_MAX,
        "var_ok": var >= VAR_MIN,
        "cost_ok": cost <= COST_TOK_MAX,
    }
    return all(checks.values()), {
        "ei":ei, "entropy":ent, "jsd":jsd, "drift":drift, "var":var, "cost":cost, "checks":checks
    }

def commit_new_state(new_text:str, metrics:dict):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    STATE_FILE.write_text(new_text, encoding="utf-8")
    run(["git","add","ET_state.md"], cwd=REPO_DIR)
    msg = (f"evolver: EI={metrics['ei']:.3f} "
           f"ent={metrics['entropy']:.2f} jsd={metrics['jsd']:.3f} drift={metrics['drift']:.3f} "
           f"var={metrics['var']:.3f} cost={metrics['cost']}")
    run(["git","commit","-q","-m",msg], cwd=REPO_DIR)
    run(["git","tag","-f", f"evolve-{ts}"], cwd=REPO_DIR)

def loop():
    ensure_git_repo()
    if not INDEX_P.exists() or not META_P.exists():
        log.error("Índice RAG não encontrado. Gere /opt/et_ultimate/index/etomega.faiss e meta.")
        return
    rag = RAG(INDEX_P, META_P)
    model_id = get_model_id()
    if not model_id:
        log.error("Nenhum modelo disponível em /v1/models.")
        return

    log.info(f"evolver iniciado | model={model_id} | repo={REPO_DIR}")
    while True:
        try:
            # “interruptor” simples: crie um arquivo stop para pausar o loop
            if (REPO_DIR/"STOP").exists():
                log.info("STOP presente; dormindo...")
                time.sleep(SLEEP_SECONDS)
                continue

            old_text = STATE_FILE.read_text(encoding="utf-8") if STATE_FILE.exists() else ""
            new_text, ei, raw = propose_new_state(model_id, rag, old_text)
            if not new_text:
                log.warning("modelo não retornou <EQUATION>... ignorando esta iteração")
                time.sleep(SLEEP_SECONDS); continue

            ok, metrics = guardrails_ok(old_text, new_text, ei)
            metrics_str = json.dumps(_json_sanitize(metrics), ensure_ascii=False, default=_json_default)
            if ok:
                commit_new_state(new_text, metrics)
                log.info(f"APLICADA nova versão | {metrics_str}")
            else:
                log.info(f"REJEITADA proposta | {metrics_str}")

        except requests.HTTPError as e:
            log.error(f"HTTP error: {e} | body={getattr(e.response,'text', '')[:300]}")
        except Exception as e:
            log.exception(f"falha no loop: {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    loop()
