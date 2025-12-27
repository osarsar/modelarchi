#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COUCHE — COMPRÉHENSION D'ÉVÉNEMENT (CPU ONLY)
Modèle : Qwen2.5-1.5B-Instruct
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# FORCER CPU (IMPORTANT)
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # désactive CUDA complètement
torch.set_num_threads(4)                  # ajuste selon ton CPU

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def build_prompt(news: str) -> str:
    return f"""
Tu es un analyste neutre.
Explique ce que signifie cette news, sans inventer.

Règles :
- N'ajoute aucune information externe
- Si une info n'est pas précisée, dis "Non précisé"
- Réponds en français

NEWS :
\"\"\"{news}\"\"\"

Explique clairement :
- Type d'événement
- Acteurs impliqués
- Pays / lieu
- Objectif de l'action
- Résumé neutre
""".strip()


def main():
    print("🧠 Chargement du modèle de compréhension (CPU uniquement)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=None
    )
    model.to("cpu")
    model.eval()

    print("📰 Donne une news :")
    news = input("> ").strip()

    if not news:
        print("❌ News vide")
        sys.exit(1)

    prompt = build_prompt(news)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=0.2,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n🧠 COMPRÉHENSION DE L’ÉVÉNEMENT\n")
    print(response.split("NEWS")[-1].strip())


if __name__ == "__main__":
    main()
