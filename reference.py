import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# CONFIG
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 80


# ===============================
# LOAD MODEL
# ===============================
print("🧠 Chargement du modèle — normalisation événementielle (TITRES SEULS)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu"
).eval()


# ===============================
# PROMPT STRICT (AMÉLIORÉ)
# ===============================
def build_prompt(title: str) -> str:
    return f"""
Tu es un extracteur institutionnel.

À partir du TITRE CI-DESSOUS, identifie l’événement administratif principal.

Extrais STRICTEMENT :

ACTION : un mot (LANCEMENT, ADOPTION, CRÉATION, FORMATION, ACCOMPAGNEMENT)
OBJET  : nom administratif court (programme, plateforme, commission, loi, décret, portail)
CIBLE  : groupe concerné (ou INCONNU)

RÈGLES ABSOLUES :
- Utilise UNIQUEMENT le titre
- PAS de phrase
- PAS d’explication
- PAS de noms propres
- PAS d’intentions ("pour", "afin de")
- Si absent → INCONNU

FORMAT EXACT :
ACTION : ...
OBJET : ...
CIBLE : ...

TITRE :
"{title}"
""".strip()


# ===============================
# PARSING LLM
# ===============================
def parse_llm_output(text: str) -> dict:
    event = {"action": "INCONNU", "objet": "INCONNU", "cible": "INCONNU"}

    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()

        if k == "action":
            event["action"] = v.upper()
        elif k == "objet":
            event["objet"] = v.lower()
        elif k == "cible":
            event["cible"] = v.lower()

    return event


# ===============================
# POST-NORMALISATION (CLÉ DU SUCCÈS)
# ===============================
def clean_event(event: dict) -> dict:
    """
    REND L'ÉVÉNEMENT COMPARABLE ET STABLE
    """

    # ACTION — verrouillée
    allowed_actions = {
        "LANCEMENT", "ADOPTION", "CRÉATION", "FORMATION", "ACCOMPAGNEMENT"
    }
    if event["action"] not in allowed_actions:
        event["action"] = "INCONNU"

    # OBJET — nettoyage fort
    obj = event["objet"]

    # Supprimer intentions
    obj = re.sub(r"\b(pour|afin de|visant à|destiné à).*", "", obj)

    # Supprimer verbes
    obj = re.sub(r"\b(sera|est|vise|permet)\b.*", "", obj)

    # Normalisations connues
    replacements = {
        "programme prioritaire": "programme de formation",
        "plateforme numérique d'enseignement à distance": "plateforme numérique éducative",
        "formation initiale dans le domaine du digital": "formation digitale",
        "portail électronique": "portail électronique gouvernemental",
        "commission ministérielle": "commission ministérielle",
        "projet de décret": "projet de décret",
        "projet de loi": "projet de loi",
        "accompagnement des startups du digital": "accompagnement startups digitales",
    }

    for k, v in replacements.items():
        if k in obj:
            obj = v
            break

    event["objet"] = obj.strip() if obj else "INCONNU"

    # CIBLE — simplification
    cible = event["cible"]
    cible = re.sub(r"\(.*?\)", "", cible)

    if len(cible) > 80:
        cible = "INCONNU"

    event["cible"] = cible.strip() if cible else "INCONNU"

    return event


# ===============================
# NORMALISATION D’UN TITRE
# ===============================
def normalize_title(doc: dict) -> dict:
    title = doc.get("title", "").strip()
    if not title:
        return None

    prompt = build_prompt(title)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "ACTION" in decoded:
        decoded = "ACTION" + decoded.split("ACTION", 1)[-1]

    event = parse_llm_output(decoded)
    event = clean_event(event)

    return {
        "date": doc.get("date_published", "").split(" Version")[0],
        "event": event
    }


# ===============================
# BATCH
# ===============================
def normalize_titles(docs: list[dict]) -> list[dict]:
    results = []

    for i, doc in enumerate(docs, 1):
        print(f"🔹 Normalisation {i}/{len(docs)}")
        ev = normalize_title(doc)
        if ev:
            results.append(ev)

    return results


# ===============================
# TEST LOCAL
# ===============================
if __name__ == "__main__":
    docs_example = [
        {
            "title": "Lancement d'un programme prioritaire de sensibilisation et de formation des nouveaux élus des collectivités territoriales",
            "date_published": "24 novembre 2021 Version Imprimable",
        },
        {
            "title": "M. Benmoussa annonce le prochain lancement d'une plateforme numérique d'enseignement à distance de l'amazigh",
            "date_published": "13 mai 2024 Version Imprimable",
        },
        {
            "title": "Formation initiale dans le domaine du digital: 20.000 bénéficiaires à l’horizon 2026 (ministre)",
            "date_published": "08 novembre 2024 Version Imprimable",
        },
        {
            "title": "Le gouvernement lance un portail électronique pour renforcer l'interaction avec les citoyens",
            "date_published": "27 décembre 2023 Version Imprimable",
        },
        {
            "title": "Une commission ministérielle sera chargée du traitement des problématiques liées au statut des fonctionnaires de l’Éducation nationale",
            "date_published": "14 novembre 2023 Version Imprimable",
        },
        {
            "title": "Adoption d'un projet de décret sur la vocation des établissements universitaires, les cycles des études supérieures et les diplômes nationaux correspondants",
            "date_published": "25 juillet 2023 Version Imprimable",
        },
        {
            "title": "Le Conseil de Gouvernement adopte un projet de loi sur l'acquisition et la mise en chantier pour la construction, la refonte ou la modification des navires de pêches",
            "date_published": "06 juillet 2023 Version Imprimable",
        },
        {
            "title": "L'accompagnement des startups du digital figure au centre des priorités du ministère de la Transition numérique (Mme Mezzour)",
            "date_published": "16 janvier 2024 Version Imprimable",
        }
    ]

    out = normalize_titles(docs_example)

    print("\n================ SORTIE NORMALISÉE =================\n")
    for o in out:
        print(o)
