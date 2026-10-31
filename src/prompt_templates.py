"""
Prompt Engineering Templates for Gen AI Mobile Recommender.
Provides strictly grounded system instructions, persona-adapted templates,
hallucination guardrails, and comparison prompts.
"""
from typing import List, Dict, Any

SYSTEM_INSTRUCTION = """You are Axiom Mobile AI, an expert, impartial smartphone technology consultant.
Your mission is to provide deeply reasoned, accurate, and personalized smartphone recommendations based strictly on the user's needs, budget, and provided device specifications.

STRICT GROUNDING & GUARDRAILS:
1. TRUTHFULNESS: Only use technical specifications present in the provided device context. NEVER hallucinate or invent specifications (e.g. refresh rate, battery capacity, camera MP, processor).
2. TRADEOFF TRANSPARENCY: Every smartphone involves compromises. Always explicitly state what the user gives up (e.g., plastic build, slow charging, lack of telephoto, heavy weight).
3. PERSONALIZATION: Address the user's specific use case directly (e.g., gaming frame stability, camera low-light performance, all-day battery life, compact one-handed usability).
4. CONTRADICTION HANDLING: If the user requests impossible combinations (e.g. $150 flagship with 8 Gen 3 and periscope zoom), explain market realities courteously and recommend the closest realistic champions.
5. FORMATTING: Use clean markdown with clear headings, bullet points, spec highlights, and a final concise verdict.
"""

def format_device_context(candidates: List[Dict[str, Any]]) -> str:
    """Formats retrieved device candidates into structured markdown for RAG prompt augmentation."""
    blocks = []
    for i, item in enumerate(candidates, 1):
        dev = item["device"]
        score = item["composite_score"]
        block = f"""### Option {i}: {dev['brand']} {dev['model']} (Fit Score: {int(score * 100)}%)
- **Price**: ${dev['price_usd']} USD / ₹{dev['price_inr']} INR
- **Processor**: {dev['processor']}
- **Display**: {dev['display']} ({dev['refresh_rate_hz']}Hz)
- **Memory & Storage**: {dev['ram_gb']}GB RAM | {dev['storage_gb']}GB Storage
- **Camera System**: {dev['camera_setup']}
- **Battery & Fast Charge**: {dev['battery_mah']}mAh | {dev['charging_watt']}W
- **OS & Software**: {dev['os']}
- **Build & Durability**: {dev['weight_g']}g | {dev['water_resistance']}
- **Target Persona**: {dev['target_persona']}
- **Verified Pros**: {dev['pros']}
- **Known Cons / Tradeoffs**: {dev['cons']}
- **User Rating**: {dev['user_rating']} / 5.0
"""
        blocks.append(block)
    return "\n".join(blocks)

def build_recommendation_prompt(query: str, candidates: List[Dict[str, Any]], relaxation_notice: str = None) -> str:
    """Constructs the full user prompt injected with retrieved RAG context."""
    devices_text = format_device_context(candidates)
    
    notice_text = ""
    if relaxation_notice:
        notice_text = f"\n*System Note: {relaxation_notice}*\n"

    prompt = f"""User Request: "{query}"
{notice_text}
Here are the top retrieved smartphone candidates from the verified database:

{devices_text}

TASK:
Provide a personalized recommendation report following this structure:
1. **Executive Recommendation**: Announce the single #1 Top Pick and state in 2 sentences why it matches the user's requirements best.
2. **Deep-Dive on Top Pick**: Detail its key hardware advantages, real-world experience, and honest compromises.
3. **Alternative / Runner-up**: Present the next best option for someone who might want different trade-offs (e.g. better software vs raw horsepower, or lower cost).
4. **Direct Feature Comparison**: A quick side-by-side markdown comparison table of the top options (Price, Processor, Screen, Battery, Key Strength).
5. **Clear Buying Verdict**: 1-2 closing sentences giving the exact deciding factor.

Ground every specification strictly in the provided data. Do not invent any specs.
"""
    return prompt

def build_comparison_prompt(device_a: Dict[str, Any], device_b: Dict[str, Any], focus_aspect: str = "General") -> str:
    """Builds a prompt for head-to-head comparison between two specific phones."""
    prompt = f"""Compare the following two smartphones head-to-head, with special attention to: {focus_aspect}.

Device A: {device_a['brand']} {device_a['model']}
- Price: ${device_a['price_usd']} USD (₹{device_a['price_inr']} INR)
- Processor: {device_a['processor']}
- Display: {device_a['display']}
- Cameras: {device_a['camera_setup']}
- Battery: {device_a['battery_mah']}mAh ({device_a['charging_watt']}W)
- OS: {device_a['os']}
- Pros: {device_a['pros']}
- Cons: {device_a['cons']}

Device B: {device_b['brand']} {device_b['model']}
- Price: ${device_b['price_usd']} USD (₹{device_b['price_inr']} INR)
- Processor: {device_b['processor']}
- Display: {device_b['display']}
- Cameras: {device_b['camera_setup']}
- Battery: {device_b['battery_mah']}mAh ({device_b['charging_watt']}W)
- OS: {device_b['os']}
- Pros: {device_b['pros']}
- Cons: {device_b['cons']}

TASK:
1. Provide a side-by-side breakdown across: Display, Camera Quality, Performance & Thermals, Battery/Charging, and Software Support.
2. State who should buy Device A and who should buy Device B.
3. Provide a definitive winner verdict based on overall value.
"""
    return prompt
