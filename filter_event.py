import os
import re
from typing import List, Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)

# Bi-encoder (multilingue, solide pour retrieval)
BI_ENCODER = "intfloat/multilingual-e5-large"

# Cross-encoder (reranker multilingue) :
# bon choix CPU : relativement léger + performant
CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Optionnel: LLM tie-breaker (seulement si borderline)
USE_LLM_TIEBREAKER = True
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K_RETRIEVE = 20

# seuils cross-encoder (à ajuster)
THRESH_ACCEPT = 0.62      # >= -> même événement
THRESH_REJECT = 0.45      # <= -> pas même événement
# entre les deux -> tie-breaker (LLM) ou rejet prudent


# =========================
# TEXT UTILS
# =========================
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_quotes(text: str) -> str:
    return text.replace("«", '"').replace("»", '"').replace("’", "'")


def make_pseudo_title_from_news(news: str, max_len: int = 220) -> str:
    """
    Convertit une phrase news -> pseudo-titre court pour faire du matching sur titres.
    (On garde uniquement l'essentiel du début)
    """
    t = normalize_spaces(strip_quotes(news))
    # enlever guillemets inutiles
    t = re.sub(r"^[-•\s]+", "", t)
    # couper à une longueur raisonnable
    return t[:max_len]


# =========================
# LOAD MODELS
# =========================
print("🧠 Chargement Bi-encoder (retrieval)...")
biencoder = SentenceTransformer(BI_ENCODER, device="cpu")

print("🧠 Chargement Cross-encoder (rerank événement)...")
cross = CrossEncoder(CROSS_ENCODER, device="cpu")

llm_tok, llm = None, None
if USE_LLM_TIEBREAKER:
    print("🧠 Chargement LLM tie-breaker (Qwen)...")
    llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="cpu").eval()


# =========================
# RETRIEVE (Bi-encoder)
# =========================
def embed_text_e5(texts: List[str]) -> torch.Tensor:
    """
    E5 recommande: "query: ..." et "passage: ..."
    Ici on utilise query/passages pour optimiser.
    """
    embs = biencoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    return embs


def retrieve_candidates(news_title: str, docs: List[Dict], top_k: int) -> List[Tuple[int, float]]:
    """
    Retourne (index_doc, score_cosine) trié desc
    """
    # embeddings titres DB
    titles = [d.get("title", "") for d in docs]
    passages = [f"passage: {t}" for t in titles]
    q = embed_text_e5([f"query: {news_title}"])[0]
    P = embed_text_e5(passages)

    # cosine (car normalisé)
    scores = (P @ q).cpu().numpy().tolist()

    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# =========================
# CROSS-ENCODER DECISION
# =========================
def cross_score_pairs(news_title: str, docs: List[Dict], idxs: List[int]) -> List[Tuple[int, float]]:
    pairs = [(news_title, docs[i]["title"]) for i in idxs]
    scores = cross.predict(pairs).tolist()
    return list(zip(idxs, scores))


# =========================
# LLM TIE-BREAKER (OUI/NON)
# =========================
def llm_same_event(news_title: str, candidate_title: str) -> bool:
    """
    Tie-breaker strict. Ne l'appelle que si nécessaire.
    """
    prompt = f"""
Tu es un classificateur strict.

TITRE A :
{news_title}

TITRE B :
{candidate_title}

Question :
Ces deux titres décrivent-ils le MÊME ÉVÉNEMENT (même action principale, même objet, même contexte),
même si la formulation est différente ?

Réponds uniquement par :
OUI
ou
NON
""".strip()

    inputs = llm_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=3, do_sample=False, eos_token_id=llm_tok.eos_token_id)

    txt = llm_tok.decode(out[0], skip_special_tokens=True).upper()
    # on cherche la dernière réponse
    return "OUI" in txt.splitlines()[-1]


# =========================
# PIPELINE FINAL
# =========================
def filter_same_event_titles(news: str, docs: List[Dict]) -> List[Dict]:
    news_title = make_pseudo_title_from_news(news)
    print("\n📰 NEWS (pseudo-titre) :", news_title)

    print("\n🔎 Étape 1 — Retrieval (Bi-encoder) ...")
    cand = retrieve_candidates(news_title, docs, TOP_K_RETRIEVE)
    print(f"➡️ {len(cand)} candidats")

    cand_idxs = [i for i, _ in cand]
    print("\n🧠 Étape 2 — Rerank / décision (Cross-encoder) ...")
    scored = cross_score_pairs(news_title, docs, cand_idxs)
    scored.sort(key=lambda x: x[1], reverse=True)

    kept = []
    for i, s in scored:
        title = docs[i]["title"]
        print(f"  score={s:.3f} | {title}")

        if s >= THRESH_ACCEPT:
            kept.append(docs[i])
            continue

        if s <= THRESH_REJECT:
            continue

        # zone grise -> tie-breaker
        if USE_LLM_TIEBREAKER:
            ok = llm_same_event(news_title, title)
            print(f"    ↳ tie-breaker LLM => {'OUI' if ok else 'NON'}")
            if ok:
                kept.append(docs[i])

    return kept


# =========================
# TEST LOCAL
# =========================
if __name__ == "__main__":
    NEWS = (
        "Le ministère de l’Éducation a annoncé le lancement d’un nouveau "
        "programme national de formation des enseignants visant à renforcer "
        "les compétences numériques dans les établissements publics."
    )

    EVENTS_DB = [
        {"date": "24 novembre 2021", "title": "Lancement d'un programme prioritaire de sensibilisation et de formation des nouveaux élus des collectivités territoriales"},
        {"date": "13 mai 2024", "title": "M. Benmoussa annonce le prochain lancement d'une plateforme numérique d'enseignement à distance de l'amazigh"},
        {"date": "08 novembre 2024", "title": "Formation initiale dans le domaine du digital: 20.000 bénéficiaires à l’horizon 2026 (ministre)"},
        {"date": "27 décembre 2023", "title": "Le gouvernement lance un portail électronique pour renforcer l'interaction avec les citoyens"},
        {"date": "14 novembre 2023", "title": "Une commission ministérielle sera chargée du traitement des problématiques liées au statut des fonctionnaires de l’Éducation nationale"},
        {"date": "25 juillet 2023", "title": "Adoption d'un projet de décret sur la vocation des établissements universitaires, les cycles des études supérieures et les diplômes nationaux correspondants"},
        {"date": "06 juillet 2023", "title": "Le Conseil de Gouvernement adopte un projet de loi sur l'acquisition et la mise en chantier pour la construction, la refonte ou la modification des navires de pêches"},
        {"date": "16 janvier 2024", "title": "L'accompagnement des startups du digital figure au centre des priorités du ministère de la Transition numérique (Mme Mezzour)"},
    ]

    kept = filter_same_event_titles(NEWS, EVENTS_DB)

    print("\n✅ TITRES CONSERVÉS (même événement) :")
    for d in kept:
        print("-", d["date"], "|", d["title"])
