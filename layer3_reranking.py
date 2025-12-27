from sentence_transformers import CrossEncoder

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
THRESHOLD_SAME_EVENT = 1.5   # seuil empirique solide

# ======================================================
# LOAD MODEL
# ======================================================
print("🧠 Chargement Cross-Encoder (Couche C)...")
cross_encoder = CrossEncoder(MODEL_NAME)


# ======================================================
# COUCHE C — RERANKING FIN
# ======================================================
def rerank_same_event(
    news_text: str,
    candidates: list[dict],
    threshold: float = THRESHOLD_SAME_EVENT
) -> list[dict]:
    """
    Compare finement la news avec chaque candidat
    et garde uniquement ceux qui parlent DU MÊME ÉVÉNEMENT
    """

    if not candidates:
        return []

    # 1. Préparer les paires (news, référence)
    pairs = []
    for c in candidates:
        ref_text = c.get("text") or c.get("title") or ""
        pairs.append((news_text, ref_text))

    # 2. Scoring cross-encoder
    scores = cross_encoder.predict(pairs)

    # 3. Filtrage strict
    kept = []
    for score, candidate in zip(scores, candidates):
        if score >= threshold:
            kept.append({
                **candidate,
                "cross_score": round(float(score), 3)
            })

    # 4. Tri final
    kept.sort(key=lambda x: x["cross_score"], reverse=True)

    return kept


if __name__ == "__main__":

    NEWS = (
        "Le ministère de l’Éducation a annoncé le lancement d’un nouveau "
        "programme national de formation des enseignants visant à renforcer "
        "les compétences numériques dans les établissements publics."
    )

    CANDIDATES_FROM_B = [
        {"date": "08 novembre 2024", "title": "Formation initiale dans le domaine du digital: 20.000 bénéficiaires"},
        {"date": "24 novembre 2021", "title": "Formation des nouveaux élus des collectivités territoriales"},
        {"date": "13 mai 2024", "title": "Plateforme numérique d’enseignement à distance de l’amazigh"},
    ]

    final_refs = rerank_same_event(NEWS, CANDIDATES_FROM_B)

    print("\n✅ ÉVÉNEMENTS CONFIRMÉS (COUCHE C):\n")
    for r in final_refs:
        print(f"{r['cross_score']} | {r['date']} | {r['title']}")
