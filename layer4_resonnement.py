from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ======================================================
# LOAD MODEL — NLI
# ======================================================
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"

print("🧠 Chargement modèle NLI (Couche D)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = ["contradiction", "neutral", "entailment"]


# ======================================================
# NLI INFERENCE
# ======================================================
def nli_score(premise: str, hypothesis: str) -> dict:
    """
    premise = référence fiable
    hypothesis = news analysée
    """
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0]

    return {
        label: round(float(probs[i]), 3)
        for i, label in enumerate(LABELS)
    }


NEWS = (
    "Le ministère de l’Éducation a annoncé la suspension immédiate "
    "du programme national de formation numérique des enseignants."
)



REFERENCE = (
    "Le ministère de l’Éducation a confirmé la poursuite et le renforcement "
    "du programme national de formation numérique des enseignants."
)




result = nli_score(REFERENCE, NEWS)

print(result)
