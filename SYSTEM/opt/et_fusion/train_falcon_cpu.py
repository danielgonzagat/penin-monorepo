#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
# Garanta que a variável que desativa LOFTQ seja setada ANTES de importar transformers/peft
os.environ.setdefault("PEFT_DISABLE_LOFTQ_IMPORT","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("HF_HOME","/opt/hf_cache")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ---------- caminhos ----------
BASE_MODEL  = "/opt/llm_base/falcon-mamba-7b"  # já baixado por você
DATA_PATH   = "/opt/et_fusion/data/lemniscata_sft.jsonl"
OUT_DIR     = "/opt/llm_finetuned/falcon-mamba-7b-lemniscata-cpu"
OUT_DIR_MERGED = OUT_DIR + "-merged"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- sanidade ----------
if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError(f"Modelo base não encontrado: {BASE_MODEL}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset não encontrado: {DATA_PATH}")

# ---------- CPU e seeds ----------
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS","4"))))
device = torch.device("cpu")
set_seed(42)

# ---------- dataset ----------
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def format_record(ex):
    instr = (ex.get("instruction","") or "").strip()
    inp   = (ex.get("input","") or "").strip()
    out   = (ex.get("output","") or "").strip()
    if inp:
        prompt = f"### Instrução:\n{instr}\n\n### Entrada:\n{inp}\n\n### Resposta:\n"
    else:
        prompt = f"### Instrução:\n{instr}\n\n### Resposta:\n"
    return {"text": prompt + out}

ds = ds.map(format_record, remove_columns=ds.column_names)

# ---------- tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
    trust_remote_code=True
)
# usa EOS como PAD se necessário
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = int(os.environ.get("MAX_LEN","1024"))

def tok(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

tok_ds = ds.map(tok, batched=True, remove_columns=["text"])

# ---------- modelo ----------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.to(device)
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))

# ---------- LoRA (sem bitsandbytes) ----------
try:
    from peft import LoraConfig, get_peft_model
except Exception as e:
    print("[FATAL] Falha ao importar peft:", e)
    sys.exit(1)

# detecta todos os nn.Linear por nome (alvo do LoRA)
import torch.nn as nn
def find_all_linear_leaf_names(m):
    names=set()
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            names.add(n.split(".")[-1])
    # Evita embeddings e cabeça de saída
    names -= {"lm_head","embed_out","wte"}
    return sorted(list(names))

target_modules = find_all_linear_leaf_names(model)
if not target_modules:
    print("[WARN] Nenhum nn.Linear encontrado para LoRA; seguindo sem filtros...")
    target_modules = None  # peft tentará mapear automaticamente se suportado

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ---------- collator ----------
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ---------- args de treino (CPU) ----------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    dataloader_num_workers=0,  # CPU estável
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=collator,
)

print("[INFO] Iniciando treino (CPU)...")
trainer.train()

print("[INFO] Salvando adapter LoRA...")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("[INFO] Tentando merge do LoRA com o modelo base...")
try:
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUT_DIR_MERGED)
    tokenizer.save_pretrained(OUT_DIR_MERGED)
    print(f"[OK] Adapter salvo em {OUT_DIR}")
    print(f"[OK] Modelo fundido salvo em {OUT_DIR_MERGED}")
except Exception as e:
    print("[WARN] Falha no merge:", repr(e))
    print("Adapters foram salvos; você pode carregar via PeftModel.from_pretrained(base, adapter).")

print("[DONE]")
