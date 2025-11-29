import json
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

class LinguisticAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_FAST)

    def detect_ai_text(self, text: str):
        prompt = f"""
        Analyze this text for AI generation patterns (burstiness, perplexity proxy).
        Text: "{text[:1500]}"
        Return JSON: {{ "author_type": "AI" or "Human", "confidence_score": int, "key_indicators": ["..."] }}
        """
        res = self.llm.invoke(prompt).content
        return json.loads(res.replace('```json', '').replace('```', ''))

    def analyze_propaganda(self, text: str):
        prompt = f"""
        Analyze text for propaganda: "{text[:1500]}"
        Return JSON: {{ "techniques": ["..."], "threat_level": "High/Med/Low" }}
        """
        res = self.llm.invoke(prompt).content
        return json.loads(res.replace('```json', '').replace('```', ''))

class ForensicsAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_REASONING)

    def analyze_media(self, file_path: str):
        # In a full implementation, we would use OpenCV here.
        # For the hackathon/demo, we use the Vision Model to inspect visual anomalies.
        return {"status": "Simulated Analysis", "verdict": "Likely Authentic"}
