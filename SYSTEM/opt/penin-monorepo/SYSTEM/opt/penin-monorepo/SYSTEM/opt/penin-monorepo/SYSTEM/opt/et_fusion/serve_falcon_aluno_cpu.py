#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, uvicorn, torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/opt/llm_finetuned/falcon-mamba-7b-lemniscata-cpu-merged"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float32)
model.eval().to("cpu")

class ChatReq(BaseModel):
    model: str = "falcon-mamba-7b-lemniscata"
    messages: List[Dict[str,str]]
    temperature: float = 0.7
    max_tokens: int = 256

app = FastAPI()

@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # junta só o conteúdo do user/system/assistant de forma simples
    prompt = ""
    for m in req.messages:
        role = m.get("role","user")
        content = m.get("content","")
        if role == "system":
            prompt += f"[SISTEMA]\n{content}\n\n"
        elif role == "assistant":
            prompt += f"[ASSISTENTE]\n{content}\n\n"
        else:
            prompt += f"[USUÁRIO]\n{content}\n\n"
    prompt += "\n[ASSISTENTE]\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(1024, req.max_tokens),
            do_sample=True,
            temperature=float(req.temperature),
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    return JSONResponse({
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{"index":0, "message":{"role":"assistant","content": text}, "finish_reason":"stop"}]
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
