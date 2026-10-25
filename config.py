import os

# Google Gemini API Settings (Free Tier optimized: 5 RPM, 20 RPD per model)
MODEL_NAME = "gemini-2.5-flash"

# Fallback models pool across available free-tier models (20 RPD each = 80 total RPD)
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3-flash-preview"
]
FALLBACK_MODEL_NAME = "gemini-3.6-flash"

# Font Settings
ARABIC_FONT_NAME = 'Arial'
ARABIC_FONT_URL = "https://raw.githubusercontent.com/kavin808/arial.ttf/master/arial.ttf"
ARABIC_FONT_FILE = "arial.ttf"

ENGLISH_FONT_NAME = 'Times-Roman'
ENGLISH_FONT_SIZE = 10
ARABIC_FONT_SIZE = 10

# Layout Settings - High-density paper utilization
PAGE_MARGIN_INCH = 0.35
COLUMN_GAP_PTS = 8

# Concurrency & Free Tier Pacing (Respecting 5 RPM and 20 RPD)
MAX_WORKERS = 1
BATCH_SIZE = 30  # Pack ~30 paragraphs per batch so average lectures only take 2-3 requests
MAX_TRANSLATION_BATCHES = 4  # Cap total batches per document to save precious daily quota
MAX_TRANSLATION_WORKERS = 1  # 1 sequential worker so requests never overlap
MIN_REQUEST_INTERVAL_SECONDS = 13.0  # 13s pacing guarantees staying below 5 RPM (4.6 RPM max)
USE_DOCUMENT_SUMMARY = False  # Keep false to save 1 request per document
ENABLE_TRANSLATION_CACHE = True  # Local cache avoids repeating requests for unchanged text
TRANSLATION_CACHE_PATH = os.path.join(os.path.dirname(__file__), "translation_cache.sqlite3")
TRANSLATION_CACHE_MAX_ENTRIES = 50000
MAX_OCR_WORKERS = 1  # Paced sequential OCR
