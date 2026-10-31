"""
Configuration settings for Gen AI Mobile Recommender.
Includes path constants, default weights, and Gemini model configurations.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if available
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Sub-directories
DATA_DIR = BASE_DIR / "data"
PROMPT_LOGS_DIR = BASE_DIR / "prompt_logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist (wrapped for Streamlit Cloud read-only filesystem)
for _dir in (DATA_DIR, PROMPT_LOGS_DIR, OUTPUTS_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # Read-only filesystem (e.g., Streamlit Cloud)

# Dataset path
CSV_PATH = DATA_DIR / "mobiles.csv"

# Gemini LLM Settings
# Primary Gemini model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Safe API key getter (never raises error, returns empty string if missing)
def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key

# Scoring Weights for Recommender (Decision Matrix)
DEFAULT_WEIGHTS = {
    "semantic_similarity": 0.45,   # Weight for user prompt semantic match
    "spec_fit": 0.35,              # Weight for hard specification alignment (RAM, battery, price)
    "user_rating": 0.20            # Weight for consumer & reviewer rating
}

# Budget tolerance (allows recommendations slightly above budget if justified by high value)
BUDGET_TOLERANCE_RATIO = 0.15      # 15% flexible ceiling
