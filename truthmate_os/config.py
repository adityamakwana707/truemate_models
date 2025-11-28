import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        # Warning only, prevents crash on import if key is missing during setup
        print("⚠️ WARNING: GOOGLE_API_KEY not found in .env")

    # Model Configuration
    # Using Flash for speed/tools, Pro for complex reasoning
    MODEL_FAST = "gemini-2.0-flash-lite-preview-02-05" 
    MODEL_REASONING = "gemini-2.0-flash-thinking-exp-01-21"

    # User Agent for Link Safety (Spoofing a real browser)
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

settings = Config()
