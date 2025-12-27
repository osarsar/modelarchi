from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.65   # seuil raisonnable production

# ======================================================
# LOAD MODEL (UNE SEULE FOIS)
# ======================================================
print("🧠 Chargement du modèle sémantique (Couche B)...")
model = SentenceTransformer(MODEL_NAME)


# ======================================================
# EMBEDDING UTILS
# ======================================================
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Transforme une liste de textes en vecteurs sémantiques
    """
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ======================================================
# COUCHE B — FILTRAGE SÉMANTIQUE
# ======================================================
def semantic_event_filter(
    news_text: str,
    events_db: list[dict],
    threshold: float = SIMILARITY_THRESHOLD
) -> list[dict]:
    """
    Compare une news avec une base d'événements et garde
    ceux qui parlent du même événement (sémantiquement)
    """

    # 1. Texte news à analyser
    query_text = news_text.strip()

    # 2. Textes de référence (titre + optionnellement contenu)
    db_texts = []
    for e in events_db:
        # 👉 tu peux adapter ici (titre seul / titre+contenu)
        text = e.get("text") or e.get("title") or ""
        db_texts.append(text.strip())

    if not db_texts:
        return []

    # 3. Embeddings
    query_vec = embed_texts([query_text])
    db_vecs = embed_texts(db_texts)

    # 4. Similarité cosinus
    scores = cosine_similarity(query_vec, db_vecs)[0]

    # 5. Filtrage
    kept = []
    for score, event in zip(scores, events_db):
        if score >= threshold:
            kept.append({
                **event,
                "semantic_score": round(float(score), 3)
            })

    # 6. Tri par pertinence
    kept.sort(key=lambda x: x["semantic_score"], reverse=True)

    return kept


if __name__ == "__main__":

    NEWS = (
        "Le ministère de l’Éducation a annoncé le lancement d’un nouveau "
        "programme national de formation des enseignants visant à renforcer "
        "les compétences numériques dans les établissements publics."
    )

    EVENTS_DB = [
        {
            "date": "08 novembre 2024",
            "title": "Formation initiale dans le domaine du digital: 20.000 bénéficiaires à l’horizon 2026",
        },
        {
            "date": "13 mai 2024",
            "title": "M. Benmoussa annonce le prochain lancement d'une plateforme numérique d'enseignement à distance de l'amazigh",
        },
        {
            "date": "06 juillet 2023",
            "title": "Le Conseil de Gouvernement adopte un projet de loi sur la pêche maritime",
        },
    ]

    results = semantic_event_filter(NEWS, EVENTS_DB)

    print("\n✅ NEWS CONSERVÉES (COUCHE B):\n")
    for r in results:
        print(f"{r['semantic_score']} | {r['date']} | {r['title']}")
