from typing import List, Dict

# ======================================================
# COUCHE E — VERDICT FINAL
# ======================================================

def final_verdict(nli_results: List[Dict]) -> Dict:
    """
    nli_results: liste de dicts {entailment, neutral, contradiction}
    """

    max_contradiction = max(r["contradiction"] for r in nli_results)
    max_entailment = max(r["entailment"] for r in nli_results)

    avg_entailment = sum(r["entailment"] for r in nli_results) / len(nli_results)
    avg_neutral = sum(r["neutral"] for r in nli_results) / len(nli_results)

    # ❌ CONTRADICTION
    if max_contradiction >= 0.65:
        return {
            "verdict": "CONTRADICTOIRE",
            "confidence": round(max_contradiction, 2),
            "explanation": "La news contient des éléments factuels en contradiction avec des sources de référence."
        }

    # ✅ CONFIRMÉE
    if avg_entailment >= 0.60:
        return {
            "verdict": "CONFIRMÉE",
            "confidence": round(avg_entailment, 2),
            "explanation": "Les faits rapportés sont confirmés par plusieurs sources concordantes."
        }

    # 🟡 PARTIELLE / MANIPULÉE
    if max_entailment >= 0.50 and avg_neutral >= 0.30:
        return {
            "verdict": "PARTIELLEMENT EXACTE",
            "confidence": round(max_entailment, 2),
            "explanation": "La news repose sur des faits réels mais omet ou déforme certains éléments."
        }

    # ⚠️ NON CONFIRMÉE
    return {
        "verdict": "NON CONFIRMÉE",
        "confidence": round(avg_neutral, 2),
        "explanation": "Aucune source fiable ne permet de confirmer ou d’infirmer clairement cette information."
    }
