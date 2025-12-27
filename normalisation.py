import re
from datetime import datetime

# ======================================================
# UTILS
# ======================================================
def clean_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    return date_str.split(" Version")[0].strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ======================================================
# ACTION DETECTION (RULE-BASED)
# ======================================================
ACTION_PATTERNS = [
    ("LANCEMENT", r"\blance|\blancement|\bannonce.*lancement"),
    ("ADOPTION", r"\badopte|\badoption"),
    ("CRÉATION", r"\bcréation|\bsera chargée|\bmise en place"),
    ("FORMATION", r"\bformation|\bformer|\bformation initiale"),
    ("ACCOMPAGNEMENT", r"\baccompagnement|\baccompagner"),
]

def extract_action(title: str) -> str:
    t = title.lower()
    for action, pattern in ACTION_PATTERNS:
        if re.search(pattern, t):
            return action
    return "INCONNU"


# ======================================================
# OBJET EXTRACTION
# ======================================================
def extract_objet(title: str) -> str:
    patterns = [
        r"programme[^,:]*",
        r"plateforme[^,:]*",
        r"portail[^,:]*",
        r"commission[^,:]*",
        r"projet de loi[^,:]*",
        r"projet de décret[^,:]*",
        r"formation[^,:]*",
    ]
    for p in patterns:
        m = re.search(p, title.lower())
        if m:
            return normalize_text(m.group(0))
    return "INCONNU"


# ======================================================
# CIBLE EXTRACTION
# ======================================================
def extract_cible(title: str) -> str:
    patterns = [
        r"élus[^,:]*",
        r"enseignants[^,:]*",
        r"citoyens[^,:]*",
        r"fonctionnaires[^,:]*",
        r"établissements[^,:]*",
        r"startups[^,:]*",
        r"langue amazighe|amazigh",
    ]
    for p in patterns:
        m = re.search(p, title.lower())
        if m:
            return normalize_text(m.group(0))
    return "INCONNU"


# ======================================================
# MAIN NORMALIZATION
# ======================================================
def normalize_event(doc: dict) -> dict:
    title = normalize_text(doc.get("title", ""))
    return {
        "date": clean_date(doc.get("date_published")),
        "event": {
            "action": extract_action(title),
            "objet": extract_objet(title),
            "cible": extract_cible(title),
        }
    }


def normalize_batch(docs: list[dict]) -> list[dict]:
    return [normalize_event(d) for d in docs if d.get("title")]


# ======================================================
# TEST LOCAL
# ======================================================
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

    out = normalize_batch(docs_example)
    for o in out:
        print(o)
