import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# CONFIG — CPU ONLY
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


SYSTEM_PROMPT = """
Tu es un analyste factuel institutionnel.

Ta tâche est de déterminer si DEUX textes parlent de LA MÊME INFORMATION FACTUELLE.

IMPORTANT :
- Tu ne dois PAS utiliser de probabilités
- Tu ne dois PAS être vague
- Tu dois raisonner comme un humain
- Deux textes ne sont "la même information" QUE SI :
  - le même fait précis est décrit
  - les mêmes acteurs principaux sont impliqués
  - le même lieu OU le même cadre est mentionné
  - le même sujet central est traité

Thème proche ≠ même information.
Compatibilité ≠ même information.

Tu dois produire :
1) Une compréhension du texte A
2) Une compréhension du texte B
3) Une comparaison factuelle point par point
4) Une décision FINALE :
   - MÊME INFORMATION
   - INFORMATION DIFFÉRENTE
   - INFORMATION CONTRADICTOIRE
"""


def build_prompt(text_a: str, text_b: str) -> str:
    return f"""
Analyse les deux textes suivants.

TEXTE A :
{text_a}

TEXTE B :
{text_b}

Réponds STRICTEMENT avec cette structure :

COMPRÉHENSION TEXTE A :
- Fait principal :
- Acteurs :
- Lieu / Cadre :
- Sujet :

COMPRÉHENSION TEXTE B :
- Fait principal :
- Acteurs :
- Lieu / Cadre :
- Sujet :

COMPARAISON FACTUELLE :
- Similarités :
- Différences :

DÉCISION FINALE :
(choisis UNE seule)
- MÊME INFORMATION
- INFORMATION DIFFÉRENTE
- INFORMATION CONTRADICTOIRE

JUSTIFICATION :
(raisonnement clair et logique)
""".strip()


def compare_like_human(text_a: str, text_b: str):
    print("🧠 Chargement du modèle...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu"
    ).eval()

    prompt = SYSTEM_PROMPT + "\n\n" + build_prompt(text_a, text_b)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.0
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n🧠 ANALYSE HUMAINE\n")
    print(answer)


if __name__ == "__main__":
    print("📰 TEXTE A :")
    text_a = input("> ")

    print("\n📰 TEXTE B :")
    text_b = input("> ")

    compare_like_human(text_a, text_b)
