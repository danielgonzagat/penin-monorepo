#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/opt/llm_finetuned/falcon-mamba-7b-lemniscata-cpu-merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()
model.to("cpu")

prompt = "Explique em poucas frases a evolução da Equação de Turing até a Lemniscata de Penin."
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
