"""
Gemini LLM Client and Robust Offline Generator.
Provides integration with the Google GenAI SDK with automatic fallback to an intelligent
offline generation engine when API keys are not provided or network is offline.
"""
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.config import get_gemini_api_key, GEMINI_MODEL, PROMPT_LOGS_DIR
from src.prompt_templates import SYSTEM_INSTRUCTION, build_recommendation_prompt, build_comparison_prompt

class GeminiRecommenderLLM:
    def __init__(self, api_key: Optional[str] = None, model_name: str = GEMINI_MODEL):
        self.api_key = api_key if api_key is not None else get_gemini_api_key()
        self.model_name = model_name
        self.client = None
        self._init_client()

    def _init_client(self):
        """Attempts to initialize the Google GenAI Client if API key is provided."""
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize Google GenAI Client: {e}. Falling back to offline engine.")
                self.client = None
        else:
            self.client = None

    def generate_recommendation(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        relaxation_notice: Optional[str] = None,
        log_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generates full recommendation text using Gemini API or Offline Mock Engine.
        Returns a dict with response text, metadata, latency, and log path.
        """
        start_time = time.time()
        prompt = build_recommendation_prompt(query, candidates, relaxation_notice)
        mode = "Offline Mock Engine (Deterministic Spec-Grounded)"
        response_text = ""

        # Attempt Gemini API call if client is available
        if self.client:
            try:
                from google import genai
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3
                    )
                )
                if response and response.text:
                    response_text = response.text
                    mode = f"Google Gemini API ({self.model_name})"
            except Exception as e:
                print(f"[Notice] Gemini API call unsuccessful ({e}). Using grounded offline engine.")
                response_text = ""

        # Fallback to offline intelligent generator if no API response
        if not response_text:
            response_text = self._offline_grounded_synthesis(query, candidates, relaxation_notice)

        latency = round(time.time() - start_time, 3)

        result_payload = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "mode": mode,
            "latency_seconds": latency,
            "top_candidates_count": len(candidates),
            "top_device": candidates[0]["device"]["model"] if candidates else None,
            "prompt_text": prompt,
            "system_instruction": SYSTEM_INSTRUCTION,
            "response": response_text
        }

        if log_to_file:
            self._save_prompt_log(result_payload)

        return result_payload

    def _offline_grounded_synthesis(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        relaxation_notice: Optional[str] = None
    ) -> str:
        """
        Intelligent, zero-hallucination offline synthesizer that creates publication-quality
        markdown recommendations using verified device specs directly from data.
        """
        if not candidates:
            return "No smartphones found matching the requested criteria. Please broaden your budget or preferences."

        top_pick = candidates[0]["device"]
        runner_up = candidates[1]["device"] if len(candidates) > 1 else None
        third = candidates[2]["device"] if len(candidates) > 2 else None

        def clean_name(d):
            if not d:
                return "N/A"
            b, m = str(d.get("brand", "")).strip(), str(d.get("model", "")).strip()
            return m if m.lower().startswith(b.lower()) else f"{b} {m}"

        top_name = clean_name(top_pick)
        runner_name = clean_name(runner_up) if runner_up else None
        third_name = clean_name(third) if third else None

        notice_md = f"> [!NOTE]\n> {relaxation_notice}\n\n" if relaxation_notice else ""

        water_prot = str(top_pick.get('water_resistance', ''))
        if water_prot.lower() in ['nan', 'none', '']:
            water_prot = "Standard Splash Resistance (No IP certification)"

        # Section 1: Executive Pick
        md = f"""{notice_md}### 🏆 Top Recommendation: {top_name}
**Verdict**: The **{top_name}** stands out as the optimal selection for your requirements. Equipped with the **{top_pick['processor']}**, **{top_pick['ram_gb']}GB RAM**, and a vivid **{top_pick['display']}**, it delivers the exact balance of performance, user satisfaction ({top_pick['user_rating']}/5.0), and price-to-performance at **${top_pick['price_usd']} USD (₹{top_pick['price_inr']} INR)**.

---

### 🔍 Deep Dive: Why the {top_pick['model']} Fits
- **Core Processing Power**: Powered by the **{top_pick['processor']}**, providing smooth multitasking and sustained responsiveness for your daily workflow and entertainment.
- **Display & Visuals**: Features a **{top_pick['display']}** with a fluid **{top_pick['refresh_rate_hz']}Hz** refresh rate for smooth scrolling and gaming.
- **Camera Capabilities**: Equipped with **{top_pick['camera_setup']}**, ensuring sharp detail and low-light versatility.
- **Battery Endurance & Speed**: Packs a **{top_pick['battery_mah']}mAh** battery with **{top_pick['charging_watt']}W** rapid charging for zero downtime.
- **Build & Form Factor**: Weighs **{top_pick['weight_g']}g** with **{water_prot}** protection.
- **Primary Advantages**: {top_pick['pros']}
- **⚠️ Honest Compromises / Tradeoffs**: {top_pick['cons']}
"""

        # Section 2: Runner-up
        if runner_up:
            md += f"""
---

### 🥈 Value Alternative / Runner-Up: {runner_name}
For buyers looking for an alternative perspective:
- **Price**: ${runner_up['price_usd']} USD (₹{runner_up['price_inr']} INR)
- **Key Hardware**: {runner_up['processor']} with {runner_up['ram_gb']}GB RAM and {runner_up['battery_mah']}mAh battery ({runner_up['charging_watt']}W).
- **Why Consider**: {runner_up['pros']}.
- **Trade-off**: {runner_up['cons']}.
"""

        # Section 3: Comparison Table
        md += f"""
---

### 📊 Side-by-Side Comparison

| Feature | {top_pick['model']} (Top Pick) | {runner_up['model'] if runner_up else 'N/A'} (Runner-up) | {third['model'] if third else 'N/A'} |
| :--- | :--- | :--- | :--- |
| **Price (USD)** | **${top_pick['price_usd']}** | ${runner_up['price_usd'] if runner_up else '-'} | ${third['price_usd'] if third else '-'} |
| **Processor** | {top_pick['processor']} | {runner_up['processor'] if runner_up else '-'} | {third['processor'] if third else '-'} |
| **RAM / Storage** | {top_pick['ram_gb']}GB / {top_pick['storage_gb']}GB | {runner_up['ram_gb'] if runner_up else '-'}GB / {runner_up['storage_gb'] if runner_up else '-'}GB | {third['ram_gb'] if third else '-'}GB / {third['storage_gb'] if third else '-'}GB |
| **Display** | {top_pick['refresh_rate_hz']}Hz | {runner_up['refresh_rate_hz'] if runner_up else '-'}Hz | {third['refresh_rate_hz'] if third else '-'}Hz |
| **Battery & Charging** | {top_pick['battery_mah']}mAh ({top_pick['charging_watt']}W) | {runner_up['battery_mah'] if runner_up else '-'}mAh ({runner_up['charging_watt'] if runner_up else '-'}W) | {third['battery_mah'] if third else '-'}mAh ({third['charging_watt'] if third else '-'}W) |
| **Primary Strength** | {top_pick['pros'][:60]}... | {runner_up['pros'][:60] if runner_up else '-'}... | {third['pros'][:60] if third else '-'}... |

---

### 💡 Final Decision Guidance
If your primary priority is maximum performance and display fidelity, choose the **{top_pick['model']}**. If you want to optimize your budget or prioritize specific alternative strengths, the **{runner_up['model'] if runner_up else top_pick['model']}** is a formidable second option.
"""
        return md

    def _save_prompt_log(self, payload: Dict[str, Any]):
        """Persists a prompt-response interaction log into prompt_logs/."""
        safe_name = "".join(c if c.isalnum() else "_" for c in payload["query"][:25]).strip("_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = PROMPT_LOGS_DIR / f"log_{timestamp_str}_{safe_name}.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except (OSError, PermissionError) as e:
            print(f"[Warning] Failed to write prompt log (read-only filesystem?): {e}")
