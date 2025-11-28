from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
import json

class ResearchAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_FAST)

    def build_consensus_graph(self, claim: str):
        """
        Generates a relationship graph (Nodes/Edges) for a specific claim.
        Designed to be fed directly into Vis.js on the frontend.
        """
        prompt = f"""
        Act as a Fact-Checking Graph Engine.
        Claim: "{claim}"
        
        Task:
        1. Identify the Main Claim.
        2. Identify 3 distinct entities/perspectives (e.g., "Scientific Consensus", "Skeptics", "Major News Outlet").
        3. Define their relationship to the claim (Supports, Disputes, Neutral).
        
        Return STRICT JSON format suitable for network visualization:
        {{
            "nodes": [
                {{ "id": 1, "label": "The Claim", "group": "claim" }},
                {{ "id": 2, "label": "Entity Name", "group": "source_support" }}
            ],
            "edges": [
                {{ "from": 2, "to": 1, "label": "confirms" }}
            ]
        }}
        """
        raw = self.llm.invoke(prompt).content
        try:
            return json.loads(raw.replace('```json', '').replace('```', ''))
        except:
            return {"nodes": [], "edges": [], "error": "Failed to parse graph"}
