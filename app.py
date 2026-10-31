"""
Streamlit Web Application for Gen AI Mobile Recommender.
Features an interactive AI Advisor, Head-to-Head Phone Comparison, Catalog Explorer,
and Prompt Engineering & Benchmark Inspector.
"""
import os
import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.config import get_gemini_api_key, DEFAULT_WEIGHTS
from src.data_loader import MobileDataLoader
from src.recommender import HybridMobileRecommender
from src.gemini_llm import GeminiRecommenderLLM
from src.prompt_templates import build_comparison_prompt, SYSTEM_INSTRUCTION

# Page Configuration
st.set_page_config(
    page_title="Axiom | Gen AI Mobile Recommender",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .phone-card {
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 18px;
        background-color: #FFFFFF;
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .phone-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tag-blue { background-color: #DBEAFE; color: #1E40AF; }
    .tag-green { background-color: #D1FAE5; color: #065F46; }
    .tag-purple { background-color: #EDE9FE; color: #5B21B6; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    loader = MobileDataLoader()
    recommender = HybridMobileRecommender(data_loader=loader)
    return loader, recommender

loader, recommender = load_system()

# Sidebar Setup
with st.sidebar:
    st.markdown("### ⚙️ System Controls")
    
    # API Key Handling (Optional)
    api_key_input = st.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value=get_gemini_api_key(),
        help="Leave blank to use the built-in deterministic offline mock engine (zero cost, zero API errors)."
    )
    
    st.divider()
    st.markdown("#### 🔍 Filter Constraints")
    
    os_choice = st.selectbox("Operating System", ["Any", "Android", "iOS"])
    
    all_brands = ["All Brands"] + loader.get_all_brands()
    brand_choice = st.selectbox("Preferred Brand", all_brands)
    
    price_bounds = loader.get_price_bounds()
    budget_slider = st.slider(
        "Max Budget (USD)",
        min_value=int(price_bounds["min_usd"]),
        max_value=int(price_bounds["max_usd"]),
        value=int(price_bounds["max_usd"]),
        step=50,
        help="Soft filter with 15% tolerance relaxation when needed."
    )
    
    st.divider()
    with st.expander("⚖️ Decision Weights (Scoring Matrix)"):
        w_semantic = st.slider("Semantic Match Weight", 0.0, 1.0, DEFAULT_WEIGHTS["semantic_similarity"], 0.05)
        w_spec = st.slider("Hardware Spec Alignment", 0.0, 1.0, DEFAULT_WEIGHTS["spec_fit"], 0.05)
        w_rating = st.slider("User Rating Weight", 0.0, 1.0, DEFAULT_WEIGHTS["user_rating"], 0.05)
        
        # Normalize weights
        total_w = w_semantic + w_spec + w_rating
        custom_weights = {
            "semantic_similarity": w_semantic / total_w,
            "spec_fit": w_spec / total_w,
            "user_rating": w_rating / total_w
        }

    st.markdown("---")
    st.caption("Axiom Mobile Recommender v1.0 | Google Gemini RAG Architecture")

# Initialize LLM with active key
llm_client = GeminiRecommenderLLM(api_key=api_key_input if api_key_input else None)

# Main Header
st.markdown('<div class="main-header">📱 Axiom | Gen AI Mobile Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Objective, grounded smartphone guidance powered by hybrid vector search and Gemini LLM synthesis.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 AI Advisor",
    "⚖️ Head-to-Head Comparison",
    "📊 Catalog Explorer",
    "🔬 Prompt & System Inspector"
])

# -------------------------------------------------------------
# TAB 1: AI Advisor
# -------------------------------------------------------------
with tab1:
    st.markdown("#### What kind of smartphone are you looking for?")
    
    # Preset Persona Quick Buttons
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    preset_query = None
    with col_p1:
        if st.button("🎮 Competitive Esports Gamer ($500)"):
            preset_query = "Competitive mobile gamer looking for 120Hz/144Hz display, Snapdragon processor, and ultra fast charging under $500"
    with col_p2:
        if st.button("🎓 Student All-Day Battery ($300)"):
            preset_query = "Value phone under $300 with huge 5000mAh+ battery life for classes, Netflix, and social media"
    with col_p3:
        if st.button("📸 Creator & Videographer (iOS)"):
            preset_query = "Professional content creator looking for best 4K video recording, optical zoom, and Apple iOS ecosystem"
    with col_p4:
        if st.button("📱 Compact Flagship (< 6.4 inch)"):
            preset_query = "Compact premium flagship smartphone under 6.4 inches with top tier cameras and water resistance"

    query_input = st.text_area(
        "Enter your requirements, priorities, and budget in plain English:",
        value=preset_query if preset_query else "Looking for a performance-focused phone under $500 with great battery life and 120Hz display",
        height=90
    )
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        find_btn = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
        
    if find_btn or preset_query:
        with st.spinner("Analyzing requirements, calculating vector similarity, and generating grounded report..."):
            rec_results = recommender.recommend(
                query=query_input,
                top_k=3,
                user_budget_usd=budget_slider if budget_slider < price_bounds["max_usd"] else None,
                preferred_brand=brand_choice,
                os_filter=os_choice,
                weights=custom_weights
            )
            
            top_candidates = rec_results["top_candidates"]
            relaxation = rec_results["relaxation_notice"]
            
            llm_response = llm_client.generate_recommendation(
                query=query_input,
                candidates=top_candidates,
                relaxation_notice=relaxation,
                log_to_file=True
            )

        # Show candidate summary pills
        st.markdown("### 🎯 Top Matched Devices")
        card_cols = st.columns(len(top_candidates))
        for idx, (c_col, candidate) in enumerate(zip(card_cols, top_candidates)):
            dev = candidate["device"]
            score = candidate["composite_score"]
            b, m = str(dev['brand']).strip(), str(dev['model']).strip()
            dev_title = m if m.lower().startswith(b.lower()) else f"{b} {m}"
            with c_col:
                st.markdown(f"""
                <div class="phone-card">
                    <span class="tag-badge tag-blue">Rank #{idx+1}</span>
                    <span class="tag-badge tag-green">Fit: {int(score*100)}%</span>
                    <h3 style="margin-top: 10px; margin-bottom: 5px;">{dev_title}</h3>
                    <h4 style="color: #2563EB; margin-top: 0;">${dev['price_usd']} <span style="font-size: 0.8rem; color: #64748B;">(₹{dev['price_inr']})</span></h4>
                    <p style="font-size: 0.85rem; color: #334155; margin-bottom: 6px;">
                        <b>⚡ Chip:</b> {dev['processor']}<br>
                        <b>📺 Display:</b> {dev['refresh_rate_hz']}Hz OLED<br>
                        <b>🔋 Battery:</b> {dev['battery_mah']}mAh ({dev['charging_watt']}W)<br>
                        <b>⭐ Rating:</b> {dev['user_rating']} / 5.0
                    </p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📑 Gen AI Recommendation Report")
        st.caption(f"Engine: {llm_response['mode']} | Response Latency: {llm_response['latency_seconds']}s")
        st.markdown(llm_response["response"])

# -------------------------------------------------------------
# TAB 2: Head-to-Head Comparison
# -------------------------------------------------------------
with tab2:
    st.markdown("#### Head-to-Head Smartphone Comparison")
    st.markdown("Compare any two devices side-by-side with AI-driven tradeoff analysis.")
    
    device_models = loader.df["model"].tolist()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        phone_1_model = st.selectbox("Select Device A", device_models, index=0)
    with col_d2:
        phone_2_model = st.selectbox("Select Device B", device_models, index=min(5, len(device_models)-1))
        
    focus_topic = st.selectbox("Comparison Priority", ["General Overall Value", "Gaming & Thermal Sustained Power", "Camera & Video Recording Quality", "Battery Endurance & Charging Speed"])
    
    if st.button("⚡ Compare Devices"):
        dev_a = loader.df[loader.df["model"] == phone_1_model].iloc[0].to_dict()
        dev_b = loader.df[loader.df["model"] == phone_2_model].iloc[0].to_dict()
        
        # Spec Table
        spec_df = pd.DataFrame({
            "Specification": [
                "Brand", "Price (USD)", "Price (INR)", "Processor", "Display & Refresh",
                "RAM / Storage", "Camera Setup", "Battery & Charging", "OS", "Weight", "User Rating"
            ],
            f"{dev_a['model']} (Device A)": [
                dev_a["brand"], f"${dev_a['price_usd']}", f"₹{dev_a['price_inr']}", dev_a["processor"],
                f"{dev_a['display']} ({dev_a['refresh_rate_hz']}Hz)", f"{dev_a['ram_gb']}GB / {dev_a['storage_gb']}GB",
                dev_a["camera_setup"], f"{dev_a['battery_mah']}mAh ({dev_a['charging_watt']}W)", dev_a["os"],
                f"{dev_a['weight_g']}g", f"{dev_a['user_rating']} / 5.0"
            ],
            f"{dev_b['model']} (Device B)": [
                dev_b["brand"], f"${dev_b['price_usd']}", f"₹{dev_b['price_inr']}", dev_b["processor"],
                f"{dev_b['display']} ({dev_b['refresh_rate_hz']}Hz)", f"{dev_b['ram_gb']}GB / {dev_b['storage_gb']}GB",
                dev_b["camera_setup"], f"{dev_b['battery_mah']}mAh ({dev_b['charging_watt']}W)", dev_b["os"],
                f"{dev_b['weight_g']}g", f"{dev_b['user_rating']} / 5.0"
            ]
        })
        st.table(spec_df.set_index("Specification"))
        
        with st.spinner("Generating comparative analysis..."):
            comp_prompt = build_comparison_prompt(dev_a, dev_b, focus_aspect=focus_topic)
            comp_result = llm_client.generate_recommendation(
                query=f"Compare {dev_a['model']} vs {dev_b['model']} focusing on {focus_topic}",
                candidates=[{"device": dev_a, "composite_score": 0.5}, {"device": dev_b, "composite_score": 0.5}],
                log_to_file=True
            )
            st.markdown("### 🏆 AI Trade-off Analysis")
            st.markdown(comp_result["response"])

# -------------------------------------------------------------
# TAB 3: Catalog Explorer
# -------------------------------------------------------------
with tab3:
    st.markdown("#### 📊 Complete Smartphone Catalog")
    st.caption(f"Displaying {len(loader.df)} devices across budget, mid-range, and flagship tiers.")
    
    catalog_display = loader.df[[
        "brand", "model", "price_usd", "price_inr", "processor",
        "ram_gb", "storage_gb", "display", "refresh_rate_hz",
        "battery_mah", "charging_watt", "user_rating", "target_persona"
    ]]
    st.dataframe(catalog_display, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: Prompt & System Inspector
# -------------------------------------------------------------
with tab4:
    st.markdown("#### 🔬 Prompt Engineering & Verification Architecture")
    
    st.markdown("##### 1. System Prompt & Hallucination Guardrails")
    st.code(SYSTEM_INSTRUCTION, language="markdown")
    
    st.markdown("##### 2. Benchmark Evaluation Metrics")
    try:
        from src.config import OUTPUTS_DIR
        metrics_file = OUTPUTS_DIR / "evaluation_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Persona Precision@1", f"{data['mean_persona_precision_at_1']}%")
            col_m2.metric("Budget Compliance", f"{data['budget_compliance_rate_pct']}%")
            col_m3.metric("Hallucination Rate", f"{data['hallucination_rate_pct']}%")
            col_m4.metric("Avg Latency", f"{data['average_latency_seconds']}s")
            
            st.markdown("##### 3. Benchmark Test Cases Table")
            st.table(pd.DataFrame(data["detailed_results"]))
    except Exception as e:
        st.info("Run `python -m src.evaluator` to refresh benchmark metrics.")
