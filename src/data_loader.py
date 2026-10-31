"""
Data Loader and Vector Representation Module.
Handles loading, validation, feature formatting, and text vectorization for smartphone records.
"""
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import CSV_PATH, DATA_DIR

class MobileDataLoader:
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = csv_path
        self.df = self._load_data()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._build_search_index()

    def _load_data(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}. Run data_generator.py first.")
        df = pd.read_csv(self.csv_path)
        # Ensure numerical types
        df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce").fillna(0)
        df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce").fillna(0)
        df["ram_gb"] = pd.to_numeric(df["ram_gb"], errors="coerce").fillna(0)
        df["storage_gb"] = pd.to_numeric(df["storage_gb"], errors="coerce").fillna(0)
        df["refresh_rate_hz"] = pd.to_numeric(df["refresh_rate_hz"], errors="coerce").fillna(60)
        df["battery_mah"] = pd.to_numeric(df["battery_mah"], errors="coerce").fillna(4000)
        df["user_rating"] = pd.to_numeric(df["user_rating"], errors="coerce").fillna(4.0)
        return df

    def _create_search_text(self, row: pd.Series) -> str:
        """Constructs an information-dense text representation for TF-IDF & semantic search."""
        parts = [
            f"Brand: {row['brand']}",
            f"Model: {row['model']}",
            f"Processor: {row['processor']}",
            f"Display: {row['display']} ({row['refresh_rate_hz']}Hz)",
            f"RAM: {row['ram_gb']}GB, Storage: {row['storage_gb']}GB",
            f"Camera: {row['camera_setup']}",
            f"Battery: {row['battery_mah']}mAh with {row['charging_watt']}W charging",
            f"OS: {row['os']}",
            f"Water resistance: {row['water_resistance']}",
            f"Persona: {row['target_persona']}",
            f"Strengths: {row['pros']}",
            f"Tradeoffs: {row['cons']}",
            f"Summary: {row['description_summary']}"
        ]
        return " | ".join(parts)

    def _build_search_index(self):
        """Constructs and caches TF-IDF vector index for fast semantic matching."""
        self.df["search_corpus"] = self.df.apply(self._create_search_text, axis=1)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_corpus"])

        # Cache vocabulary and metadata index (skip on read-only filesystem like Streamlit Cloud)
        cache_file = DATA_DIR / "mobile_embeddings.json"
        try:
            metadata = [
                {
                    "id": row["id"],
                    "model": row["model"],
                    "brand": row["brand"],
                    "price_usd": float(row["price_usd"]),
                    "persona": row["target_persona"]
                }
                for _, row in self.df.iterrows()
            ]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"total_devices": len(metadata), "devices": metadata}, f, indent=2)
        except OSError:
            pass  # Read-only filesystem (e.g., Streamlit Cloud)

    def get_all_brands(self) -> List[str]:
        return sorted(self.df["brand"].dropna().unique().tolist())

    def get_price_bounds(self) -> Dict[str, float]:
        return {
            "min_usd": float(self.df["price_usd"].min()),
            "max_usd": float(self.df["price_usd"].max()),
            "min_inr": float(self.df["price_inr"].min()),
            "max_inr": float(self.df["price_inr"].max()),
        }

    def get_phone_by_id(self, phone_id: str) -> Optional[Dict[str, Any]]:
        matches = self.df[self.df["id"] == phone_id]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None

    def filter_phones(
        self,
        max_price_usd: Optional[float] = None,
        max_price_inr: Optional[float] = None,
        min_ram_gb: Optional[int] = None,
        min_battery_mah: Optional[int] = None,
        min_refresh_rate: Optional[int] = None,
        preferred_brands: Optional[List[str]] = None,
        os_preference: Optional[str] = None
    ) -> pd.DataFrame:
        filtered = self.df.copy()

        if max_price_inr is not None and max_price_inr > 0:
            filtered = filtered[filtered["price_inr"] <= max_price_inr]
        elif max_price_usd is not None and max_price_usd > 0:
            filtered = filtered[filtered["price_usd"] <= max_price_usd]

        if min_ram_gb is not None and min_ram_gb > 0:
            filtered = filtered[filtered["ram_gb"] >= min_ram_gb]

        if min_battery_mah is not None and min_battery_mah > 0:
            filtered = filtered[filtered["battery_mah"] >= min_battery_mah]

        if min_refresh_rate is not None and min_refresh_rate > 60:
            filtered = filtered[filtered["refresh_rate_hz"] >= min_refresh_rate]

        if preferred_brands and len(preferred_brands) > 0:
            filtered = filtered[filtered["brand"].isin(preferred_brands)]

        if os_preference and os_preference.lower() != "any":
            if "ios" in os_preference.lower() or "apple" in os_preference.lower():
                filtered = filtered[filtered["os"].str.contains("iOS", case=False, na=False)]
            elif "android" in os_preference.lower():
                filtered = filtered[filtered["os"].str.contains("Android|One UI|OxygenOS|HyperOS|Nothing|Hello|Funtouch|Realme|MagicOS|XOS|My UX", case=False, na=False)]

        return filtered
