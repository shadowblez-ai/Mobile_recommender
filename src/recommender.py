"""
Hybrid Recommendation Engine for Smartphone Selection.
Combines hard-constraint filtering, TF-IDF vector similarity, domain-specific heuristic boosts,
and composite scoring to select the top candidate devices for Gen AI RAG synthesis.
"""
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import MobileDataLoader
from src.config import DEFAULT_WEIGHTS, BUDGET_TOLERANCE_RATIO

class HybridMobileRecommender:
    def __init__(self, data_loader: Optional[MobileDataLoader] = None):
        self.loader = data_loader or MobileDataLoader()

    def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Extracts key intents, budgets, and hardware preferences from natural language query."""
        q_lower = query.lower()
        extracted: Dict[str, Any] = {
            "budget_usd": None,
            "budget_inr": None,
            "os_preference": "any",
            "is_gaming": False,
            "is_camera": False,
            "is_battery": False,
            "is_compact": False,
            "is_budget_conscious": False,
            "brand_mentions": []
        }

        # 1. Shorthand 'k' e.g. "under 20k", "below 30k" (typical Indian subcontinent notation)
        k_match = re.search(r'(?:under|below|budget of|within|<)\s*(\d{1,2})\s*k\b', q_lower)
        if k_match:
            val_k = float(k_match.group(1)) * 1000
            extracted["budget_inr"] = val_k
            extracted["budget_usd"] = val_k / 83.0

        # 2. Explicit USD pattern e.g. "$500", "500 usd", "under $400"
        usd_match = re.search(r'(?:\$\s*(\d+))|(?:(?:under|below|within|<)\s*\$\s*(\d+))|(?:(\d+)\s*(?:usd|dollars))', q_lower)
        if usd_match and extracted["budget_usd"] is None:
            raw_usd = usd_match.group(1) or usd_match.group(2) or usd_match.group(3)
            if raw_usd:
                extracted["budget_usd"] = float(raw_usd)

        # 3. Numeric budget detection (e.g. "under 20000", "under 15000 inr", "rs 25000")
        num_match = re.search(r'(?:under|below|within|budget of|<|rs\.?|inr|₹)\s*(\d{1,3}(?:,\d{3})+|\d{3,6})\s*(?:rs|inr|rupees)?', q_lower)
        if num_match and extracted["budget_usd"] is None:
            raw_val = float(num_match.group(1).replace(",", ""))
            # If >= 2500 or mentions rupee words, it's INR
            if raw_val >= 2500 or any(cur in q_lower for cur in ["rs", "inr", "₹", "rupee"]):
                extracted["budget_inr"] = raw_val
                extracted["budget_usd"] = round(raw_val / 83.0, 1)
            else:
                extracted["budget_usd"] = raw_val

        # OS detection
        if "ios" in q_lower or "iphone" in q_lower or "apple" in q_lower:
            extracted["os_preference"] = "iOS"
        elif "android" in q_lower:
            extracted["os_preference"] = "Android"

        # Persona & feature intents
        if any(w in q_lower for w in ["game", "gaming", "fps", "esports", "genshin", "bgmi", "pubg"]):
            extracted["is_gaming"] = True
        if any(w in q_lower for w in ["camera", "photo", "photographer", "video", "4k", "zoom", "portrait", "vlog"]):
            extracted["is_camera"] = True
        if any(w in q_lower for w in ["battery", "endurance", "mah", "2-day", "all-day", "long battery"]):
            extracted["is_battery"] = True
        if any(w in q_lower for w in ["compact", "small", "pocket", "one-hand", "lightweight"]):
            extracted["is_compact"] = True
        if any(w in q_lower for w in ["budget", "cheap", "affordable", "value for money", "student"]):
            extracted["is_budget_conscious"] = True

        # Brand mentions
        for brand in self.loader.get_all_brands():
            if brand.lower() in q_lower:
                extracted["brand_mentions"].append(brand)

        return extracted

    def recommend(
        self,
        query: str,
        top_k: int = 3,
        user_budget_usd: Optional[float] = None,
        preferred_brand: Optional[str] = None,
        os_filter: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        intents = self.parse_query_intent(query)
        effective_budget_usd = user_budget_usd or intents["budget_usd"]
        effective_budget_inr = intents.get("budget_inr")
        effective_os = os_filter if (os_filter and os_filter != "any") else intents["os_preference"]

        preferred_brands = []
        if preferred_brand and preferred_brand != "All Brands":
            preferred_brands = [preferred_brand]
        elif intents["brand_mentions"]:
            preferred_brands = intents["brand_mentions"]

        w = weights or DEFAULT_WEIGHTS

        # 1. Candidate Filtering with INR or USD support
        candidates = self.loader.filter_phones(
            max_price_usd=effective_budget_usd if effective_budget_inr is None else None,
            max_price_inr=effective_budget_inr,
            preferred_brands=preferred_brands if preferred_brands else None,
            os_preference=effective_os if effective_os != "any" else None
        )

        relaxed_notice = None
        # Graceful relaxation if strict filter is empty or too constrained
        if len(candidates) < 2 and (effective_budget_usd is not None or effective_budget_inr is not None):
            relaxed_inr = effective_budget_inr * (1.0 + BUDGET_TOLERANCE_RATIO) if effective_budget_inr else None
            relaxed_usd = effective_budget_usd * (1.0 + BUDGET_TOLERANCE_RATIO) if effective_budget_usd else None
            candidates = self.loader.filter_phones(
                max_price_usd=relaxed_usd if relaxed_inr is None else None,
                max_price_inr=relaxed_inr,
                preferred_brands=preferred_brands if preferred_brands else None,
                os_preference=effective_os if effective_os != "any" else None
            )
            if effective_budget_inr:
                relaxed_notice = f"Strict budget (₹{effective_budget_inr:.0f}) had limited matches; extended by {int(BUDGET_TOLERANCE_RATIO*100)}% to ₹{relaxed_inr:.0f} to surface high-value options."
            else:
                relaxed_notice = f"Strict budget (${effective_budget_usd:.0f}) had limited matches; extended by {int(BUDGET_TOLERANCE_RATIO*100)}% to ${relaxed_usd:.0f} to surface high-value options."

        if len(candidates) == 0:
            # Fall back to entire dataset with brand/OS if possible
            candidates = self.loader.filter_phones(
                preferred_brands=preferred_brands if preferred_brands else None,
                os_preference=effective_os if effective_os != "any" else None
            )
            if len(candidates) == 0:
                candidates = self.loader.df.copy()
            relaxed_notice = "No devices met all strict constraints. Recommending best overall devices matching your functional requirements."

        # 2. Vector Cosine Similarity
        query_vector = self.loader.vectorizer.transform([query])
        candidate_indices = candidates.index.tolist()
        candidate_tfidf = self.loader.tfidf_matrix[candidate_indices]
        similarities = cosine_similarity(query_vector, candidate_tfidf).flatten()

        # 3. Composite Scoring
        scores = []
        for idx, (_, row) in enumerate(candidates.iterrows()):
            semantic_score = float(similarities[idx])

            # Rating score [0, 1]
            rating_score = float(row["user_rating"]) / 5.0

            # Spec alignment heuristic
            spec_score = 0.5  # Baseline

            if intents["is_gaming"]:
                if row["refresh_rate_hz"] >= 144:
                    spec_score += 0.25
                elif row["refresh_rate_hz"] >= 120:
                    spec_score += 0.15
                if any(chip in str(row["processor"]) for chip in ["Gen 3", "Gen 2", "A18", "9300", "8s Gen 3"]):
                    spec_score += 0.20

            if intents["is_camera"]:
                if any(cam in str(row["camera_setup"]) for cam in ["Periscope", "Telephoto", "ZEISS", "Leica", "Hasselblad", "1-inch"]):
                    spec_score += 0.25
                if "OIS" in str(row["camera_setup"]):
                    spec_score += 0.10

            if intents["is_battery"]:
                if row["battery_mah"] >= 5500:
                    spec_score += 0.30
                elif row["battery_mah"] >= 5000:
                    spec_score += 0.15
                if row["charging_watt"] >= 65:
                    spec_score += 0.10

            if intents["is_compact"]:
                if "6.1" in str(row["display"]) or "6.2" in str(row["display"]) or "6.3" in str(row["display"]) or "4.7" in str(row["display"]):
                    spec_score += 0.25
                if row["weight_g"] <= 180:
                    spec_score += 0.15

            # Budget efficiency bonus
            if effective_budget_inr is not None and effective_budget_inr > 0:
                if row["price_inr"] <= effective_budget_inr:
                    savings_ratio = (effective_budget_inr - row["price_inr"]) / effective_budget_inr
                    spec_score += min(0.15, savings_ratio * 0.2)
                else:
                    spec_score -= 0.15
            elif effective_budget_usd is not None and effective_budget_usd > 0:
                if row["price_usd"] <= effective_budget_usd:
                    savings_ratio = (effective_budget_usd - row["price_usd"]) / effective_budget_usd
                    spec_score += min(0.15, savings_ratio * 0.2)
                else:
                    spec_score -= 0.15

            spec_score = max(0.0, min(1.0, spec_score))

            # Final Composite Score
            final_composite = (
                w["semantic_similarity"] * semantic_score +
                w["spec_fit"] * spec_score +
                w["user_rating"] * rating_score
            )

            scores.append({
                "index": idx,
                "composite_score": round(final_composite, 4),
                "semantic_score": round(semantic_score, 4),
                "spec_score": round(spec_score, 4),
                "rating_score": round(rating_score, 4),
                "device": row.to_dict()
            })

        # Rank by composite score descending
        scores.sort(key=lambda x: x["composite_score"], reverse=True)
        top_candidates = scores[:top_k]

        return {
            "query": query,
            "intents": intents,
            "top_candidates": top_candidates,
            "relaxation_notice": relaxed_notice,
            "total_candidates_evaluated": len(candidates)
        }
