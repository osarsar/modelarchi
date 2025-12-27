# ==========================================================
# COUCHE A — NORMALISATION ÉVÉNEMENTIELLE (PRODUCTION)
# ==========================================================
# - Pas de LLM
# - Pas de hasard
# - 100 % déterministe
# ==========================================================

import re
from typing import Dict

# ==========================================================
# A1 — NETTOYAGE & STANDARDISATION
# ==========================================================

def clean_text(text: str) -> str:
    """
    Nettoyage strict du texte
    """
    text = text.lower()
    text = re.sub(r"\(.*?\)", " ", text)        # supprimer parenthèses
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# A2 — DICTIONNAIRES CANONIQUES
# ==========================================================

ACTION_MAP = {
    "lance": "LANCEMENT",
    "lancement": "LANCEMENT",
    "annonce": "LANCEMENT",
    "déploie": "LANCEMENT",

    "adopte": "ADOPTION",
    "adoption": "ADOPTION",

    "création": "CRÉATION",
    "mise en place": "CRÉATION",
    "commission": "CRÉATION",

    "formation": "FORMATION",
    "former": "FORMATION",
    "renforcement": "FORMATION",

    "accompagnement": "ACCOMPAGNEMENT",
    "accompagner": "ACCOMPAGNEMENT",
}

OBJET_PATTERNS = [
    (r"programme[^ ]* formation", "programme formation"),
    (r"programme[^ ]*", "programme"),
    (r"plateforme[^ ]*", "plateforme"),
    (r"portail[^ ]*", "portail"),
    (r"commission[^ ]*", "commission"),
    (r"projet de loi[^ ]*", "projet loi"),
    (r"projet de décret[^ ]*", "projet décret"),
    (r"formation[^ ]*", "formation"),
]

CIBLE_PATTERNS = [
    (r"enseignants?", "enseignants"),
    (r"élus?", "élus"),
    (r"citoyens?", "citoyens"),
    (r"fonctionnaires?", "fonctionnaires"),
    (r"établissements?", "établissements"),
    (r"startups?", "startups"),
    (r"amazigh|amazighe", "amazigh"),
]


# ==========================================================
# A3 — EXTRACTION SYMBOLIQUE
# ==========================================================

def extract_action(text: str) -> str:
    for k, v in ACTION_MAP.items():
        if k in text:
            return v
    return "INCONNU"


def extract_objet(text: str) -> str:
    for pattern, canon in OBJET_PATTERNS:
        if re.search(pattern, text):
            return canon
    return "INCONNU"


def extract_cible(text: str) -> str:
    for pattern, canon in CIBLE_PATTERNS:
        if re.search(pattern, text):
            return canon
    return "INCONNU"


# ==========================================================
# A4 — NORMALISATION FINALE
# ==========================================================

def normalize_news(news_text: str) -> Dict[str, str]:
    """
    NORMALISATION ÉVÉNEMENTIELLE FINALE
    """
    clean = clean_text(news_text)

    action = extract_action(clean)
    objet = extract_objet(clean)
    cible = extract_cible(clean)

    # canonisation finale
    if action == "FORMATION" and objet == "programme":
        objet = "programme formation"

    if objet == "formation" and cible != "INCONNU":
        objet = f"formation {cible}"

    return {
        "action": action,
        "objet": objet,
        "cible": cible
    }


# ==========================================================
# TEST LOCAL
# ==========================================================

if __name__ == "__main__":

    NEWS = (
        "M. Akhannouch réaffirme l'engagement du gouvernement pour un Etat social fort"
    )

    event = normalize_news(NEWS)

    print("\n🧠 NORMALISATION — COUCHE A\n")
    print(event)
