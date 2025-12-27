import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===============================
# CONFIG
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # enlever si GPU Toubkal
torch.set_num_threads(4)

DOCS_PATH = "../index/map_docs.npy"

EMBED_MODEL = "intfloat/multilingual-e5-large"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TOP_K = 20          # on prend large
FINAL_KEEP = 8      # après raisonnement
SIM_THRESHOLD = 0.65


# ===============================
# LOAD
# ===============================
print("\n🧠 [INIT] Chargement des modèles...")

print("🔹 Chargement encodeur sémantique (E5)")
encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
print("   ✔ Encodeur chargé")

print("🔹 Chargement tokenizer LLM")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

print("🔹 Chargement modèle LLM (raisonnement humain)")
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="cpu"
).eval()
print("   ✔ LLM prêt")

print("🔹 Chargement base de données titres")
docs = np.load(DOCS_PATH, allow_pickle=True)
print(f"   ✔ {len(docs)} documents chargés")


# ===============================
# STEP 1 — SEMANTIC FILTER
# ===============================
def retrieve_candidates(news):
    print("\n🔎 [STEP 1] Recherche sémantique sur les TITRES")
    print("📰 News entrée :", news)

    q = encoder.encode(
        f"query: {news}",
        normalize_embeddings=True
    )

    candidates = []

    for idx, d in enumerate(docs):
        title = d.get("title", "")
        if not title:
            continue

        emb = encoder.encode(
            f"passage: {title}",
            normalize_embeddings=True
        )

        score = float(np.dot(q, emb))

        if score >= SIM_THRESHOLD:
            print(f"   ➜ MATCH POTENTIEL [{idx}]")
            print(f"      Titre : {title}")
            print(f"      Score : {round(score, 3)}")
            candidates.append((score, title))

    candidates.sort(reverse=True, key=lambda x: x[0])

    print(f"\n📊 Total candidats retenus (score ≥ {SIM_THRESHOLD}) : {len(candidates)}")
    print(f"📌 Top {TOP_K} conservés pour raisonnement humain")

    return candidates[:TOP_K]


# ===============================
# STEP 2 — HUMAN-LIKE REASONING
# ===============================
def same_information(news, title):
    print("\n🧠 [STEP 2] Raisonnement humain (LLM)")
    print("🔹 Comparaison :")
    print("   A (news)  :", news)
    print("   B (titre) :", title)

    prompt = f"""
Question très précise :

Titre A (news reçue) :
"{news}"

Titre B (base de données) :
"{title}"

Question :
Ces deux titres parlent-ils de la MÊME INFORMATION FACTUELLE ?

Réponds uniquement par :
- OUI : si c'est exactement la même information
- NON : sinon

Réponse :
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        out = llm.generate(
            **inputs,
            max_new_tokens=5
        )

    answer = tokenizer.decode(out[0], skip_special_tokens=True).upper()

    decision = "OUI" if "OUI" in answer else "NON"

    print("🧾 Réponse brute LLM :", answer)
    print("➡️ Décision finale :", decision)

    return decision == "OUI"


# ===============================
# MAIN
# ===============================
def main():
    print("\n📰 [INPUT] Donne une news :")
    news = input("> ").strip()

    print("\n🚀 DÉMARRAGE PIPELINE DE LOCALISATION DE L’INFORMATION")
    print("=" * 80)

    # ---- STEP 1
    candidates = retrieve_candidates(news)

    if not candidates:
        print("\n❌ Aucun titre candidat trouvé")
        return

    print("\n🧠 [STEP 2] Validation humaine des candidats\n")

    confirmed = []

    for i, (score, title) in enumerate(candidates, 1):
        print(f"\n🔍 CANDIDAT #{i}")
        print(f"📌 Titre : {title}")
        print(f"📊 Score sémantique : {round(score, 3)}")

        ok = same_information(news, title)

        if ok:
            print("✅ CONFIRMÉ : même information")
            confirmed.append(title)
        else:
            print("❌ REJETÉ : information différente")

        print("-" * 80)

        if len(confirmed) >= FINAL_KEEP:
            print("⚠️ Limite atteinte — arrêt anticipé")
            break

    print("\n📌 TITRES DE RÉFÉRENCE (OÙ L’INFO EST PARLÉE)")
    print("=" * 80)

    if not confirmed:
        print("❌ Aucun titre validé comme référence fiable")
        return

    for t in confirmed:
        print("•", t)

    print("\n✅ FIN DU PROCESSUS")


if __name__ == "__main__":
    main()
