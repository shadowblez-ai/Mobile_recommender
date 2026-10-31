[README.md](https://github.com/user-attachments/files/34213681/README.md)
# 📱 Axiom | Gen AI Mobile Recommender System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![LLM](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Status](https://img.shields.io/badge/Status-Complete%20Final%20Submission-success.svg)]()

> **Axiom** is an objective, grounded smartphone recommendation system combining **Hybrid Retrieval-Augmented Generation (RAG)**, multi-criteria mathematical scoring, and Google Gemini LLMs to eliminate spec confusion, hallucinations, and marketing bias in tech purchasing.

---

## 📋 Submission Deliverables Summary

| Deliverable | Description | Location / Status |
| :--- | :--- | :--- |
| **Deliverable 1: Codebase Archive (.zip)** | Complete project folder with pure Python code, datasets, prompt logs, outputs, presentation, and README.md. **Strictly zero API keys included.** | `Mobile_Recommender_Project.zip` |
| **Deliverable 2: Final Presentation (.pptx)** | Complete 12-slide presentation covering: problem & audience, key decisions/weights, live demo structure, bug case study, and roadmap. | `Final_Presentation_Mobile_Recommender.pptx` & `PRESENTATION_GUIDE.md` |
| **Deliverable 3: Live Video Demo Link** | Timed 2–3 minute video demonstration script with second-by-second screen walkthrough and voiceover instructions for Google Drive. | `VIDEO_DEMO_SCRIPT.md` |

---

## 🧠 System Architecture & Workflow

Axiom uses a **Two-Stage Hybrid RAG Pipeline** designed to provide 100% truthful, verifiable recommendations:

```
[ User Natural Language Query ]
               │
               ▼
   1. Intent & Currency Parsing (USD / INR / 'k' notation, OS, Persona)
               │
               ▼
   2. Hard Spec Filtering & Graceful Relaxation (Budget + 15% elasticity)
               │
               ▼
   3. TF-IDF Semantic Vector Similarity (2-gram cosine similarity)
               │
               ▼
   4. Multi-Criteria Composite Scoring Matrix
      Fit Score = 0.45*SemanticMatch + 0.35*SpecAlignment + 0.20*UserRating
               │
               ▼
   5. Strict RAG Prompt Augmentation (Injected verified JSON context)
               │
               ▼
   6. Google Gemini LLM / Deterministic Offline Synthesizer
               │
               ▼
[ Structured Recommendation Report with Trade-offs & Comparison Table ]
```

---

## 🎯 Key Design Decisions & Scoring Matrix

1. **Scoring Weights (Decision Matrix)**:
   - **Semantic Similarity (45%)**: Understands natural language nuance (e.g. *"lightweight with clean software"* or *"esports high FPS"*).
   - **Hardware Spec Alignment (35%)**: Direct deterministic bonus for target refresh rates (120Hz/144Hz), battery capacities (5000mAh+), and flagship chipsets.
   - **User Rating (20%)**: Normalized real-world consumer and reviewer satisfaction.
2. **15% Budget Elasticity Rule**: A rigid budget cap often excludes a device priced $10 higher that offers 2x the longevity and double the battery. Axiom surfaces such options with a clear transparency badge.
3. **Dual Currency Engine**: Seamlessly interprets both USD (`$500`, `under 400 usd`) and Indian Rupees (`under 20000`, `under 20k`, `rs 25000`).

---

## 🐞 Failure Case Study: "One Thing the AI Got Wrong and How We Caught It"

- **The Incident**: During initial benchmark evaluation with the query *"Best phone under 20000 with 120Hz AMOLED and fast charging"*, the recommender returned a **$1,499 flagship phone**!
- **Root Cause**: The regex entity parser captured `20000` into the default USD budget variable, causing the engine to believe the user had a **$20,000 USD budget**.
- **How We Caught It**: Automated benchmark testing across test personas flagged a drop in budget compliance on unprefixed numerical inputs.
- **The Permanent Fix**: Implemented a dual-currency classification rule: any unprefixed budget $\ge 2,500$ is recognized as INR and mapped to dedicated INR database columns.
- **Result**: The query now accurately recommends the **Samsung Galaxy M34 5G** (₹16,999) and **Realme Narzo 70 Pro 5G** (₹19,999) with 100% precision.

---

## 📊 Benchmark Evaluation Scorecard

Evaluated across 6 diverse personas and edge cases (`outputs/evaluation_report.md`):

| Evaluation Metric | Measured Result | Benchmark Target | Verdict |
| :--- | :--- | :--- | :--- |
| **Persona Precision@1** | **100.0%** | > 85.0% | 🟢 Exceeded |
| **Budget Compliance Rate** | **100.0%** | > 90.0% | 🟢 Exceeded |
| **Hallucination Rate** | **0.0%** | < 2.0% | 🟢 Zero Hallucination |
| **Average Response Latency** | **0.05s** (Offline) / **1.42s** (Gemini) | < 2.5s | 🟢 Ultra Fast |
| **Tested Personas** | **6 Scenarios** (incl. Contradictions) | - | 🟢 Complete |

---

## 📁 Repository Directory Structure

```
Mobile_recommendation/
├── data/
│   ├── mobiles.csv                      # 60+ smartphone models across all brands, tiers & specs
│   └── mobile_embeddings.json           # Cached vector space metadata
├── src/
│   ├── __init__.py                      # Package initialization
│   ├── config.py                        # Path constants, default weights, safe API handling
│   ├── data_loader.py                   # CSV loader, TF-IDF vectorizer, and constraint filters
│   ├── recommender.py                   # Hybrid engine: intent parsing, cosine similarity & scoring
│   ├── prompt_templates.py              # System prompt, anti-hallucination guardrails & schemas
│   ├── gemini_llm.py                    # Gemini API client with smart offline fallback synthesizer
│   └── evaluator.py                     # Quantitative evaluation suite & benchmark generator
├── prompt_logs/
│   ├── log_*.json                       # Individual JSON prompt-response interaction logs
│   └── all_prompt_logs_summary.md       # Architectural prompt analysis & persona breakdowns
├── outputs/
│   ├── evaluation_metrics.json          # Machine-readable evaluation scorecard
│   └── evaluation_report.md             # Markdown benchmark evaluation report
├── app.py                               # Interactive Streamlit Web Application
├── run_cli.py                           # Command-line interface for fast terminal testing
├── create_presentation.py               # Script generating the 12-slide PowerPoint presentation
├── Final_Presentation_Mobile_Recommender.pptx # Deliverable 2 Presentation File
├── PRESENTATION_GUIDE.md                # Slide-by-slide transcript and speaking notes
├── VIDEO_DEMO_SCRIPT.md                 # Deliverable 3 Video Recording Script (2-3 mins)
├── requirements.txt                     # Pinned project dependencies
├── .env.example                         # Safe configuration template (NO secret keys)
├── README.md                            # Comprehensive project documentation
└── Mobile_Recommender_Project.zip       # Final Deliverable 1 Codebase Archive
```

---

## 🚀 Quickstart & How to Run

### 1. Prerequisites & Virtual Environment Setup
```bash
# Clone or navigate to the project directory
cd Mobile_recommendation

# Activate existing virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Or create a new virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Optional: Configure Gemini API Key
To use live Google Gemini API calls, set your key in `.env` (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env and insert: GEMINI_API_KEY=your_actual_key_here
```
> **Note**: An API key is **100% optional**! If omitted, Axiom uses its built-in deterministic, spec-grounded offline synthesizer with zero crashes or API rate limits.

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 4. Run Interactive CLI (Terminal Mode)
```bash
# General query
python run_cli.py --query "Best gaming phone under $500 with 144Hz"

# Indian Rupee query
python run_cli.py --query "Best phone under 20000 with 120Hz AMOLED and fast charging"
```

### 5. Run the Automated Evaluation Benchmark
```bash
python -m src.evaluator
```
Generates updated metrics in `outputs/evaluation_metrics.json` and `outputs/evaluation_report.md`.

### 6. Regenerate PowerPoint Presentation
```bash
python create_presentation.py
```
Generates `Final_Presentation_Mobile_Recommender.pptx`.

---

## 🔒 Security & Privacy Notice
In compliance with submission guidelines, **no API keys, secrets, credentials, or private tokens are contained in this codebase or archive**. All environment settings are managed via `.env.example` and safe fallback handlers.
