import os
import io
import base64
from PIL import Image, ImageChops, ImageEnhance
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from config import settings
import json

class DeepfakeAgent:
    def __init__(self):
        # Using REASONING model (Flash Thinking) for deepfake detection as it requires complex visual analysis
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_REASONING, temperature=0.2)

    def generate_ela(self, image_path: str, quality: int = 90):
        """
        Generates Error Level Analysis (ELA) image to detect compression anomalies.
        AI images often have uniform noise; Real edits have high contrast edges.
        """
        original = Image.open(image_path).convert('RGB')
        
        # Save compressed temporary version in memory
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Calculate difference
        ela_im = ImageChops.difference(original, resaved)
        
        # Enhance brightness to make artifacts visible
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        # Save ELA to disk for debugging/frontend
        ela_path = image_path + "_ela.jpg"
        ela_im.save(ela_path)
        return ela_path

    def get_metadata_report(self, image_path: str):
        """Extracts EXIF to find 'Software' tags commonly left by AI tools"""
        try:
            img = Image.open(image_path)
            exif = img.getexif()
            report = {"has_exif": False, "software": "Unknown", "camera": "Unknown"}
            
            if exif:
                report["has_exif"] = True
                # Tags: 305=Software, 271=Make, 272=Model
                report["software"] = exif.get(305, "Unknown")
                report["camera"] = f"{exif.get(271, '')} {exif.get(272, '')}".strip()
                
            return report
        except:
            return {"error": "Could not read metadata"}

    def analyze(self, image_path: str):
        """
        Runs the full Hybrid Forensic Pipeline
        1. Metadata Scan
        2. ELA Generation
        3. Multi-Modal LLM Analysis (Original + ELA)
        """
        print(f"🕵️ Deepfake Agent: Analyzing {image_path}...")
        
        # 1. Metadata Layer
        meta_report = self.get_metadata_report(image_path)
        
        # 2. ELA Layer
        ela_path = self.generate_ela(image_path)
        
        # 3. Vision Layer (The Judge)
        # We send BOTH images to Gemini
        
        # Prepare Prompt
        prompt_text = f"""
        Act as a Senior Digital Forensics Expert.
        You are provided with two images:
        1. The ORIGINAL image to analyze for visual artifacts.
        2. The ELA (Error Level Analysis) map of the same image.

        METADATA REPORT: {json.dumps(meta_report)}

        TASK:
        1. Analyze the ORIGINAL for: 
           - Inconsistent lighting/shadows.
           - unnatural skin textures (too smooth/plastic).
           - Deformed hands, eyes, or background details.
           - Text rendering issues.
        
        2. Analyze the ELA MAP for:
           - Uniform noise (indicates possible AI generation).
           - High-contrast patches (indicates manipulation/Photoshop).
           - Consistent compression blocks (indicates real camera).

        VERDICT:
        Is this image Real or AI/Fake? 
        Provide a confidence score (0-100) and a detailed technical explanation.
        
        Return JSON ONLY:
        {{
            "verdict": "REAL" or "FAKE/AI",
            "confidence_score": int,
            "visual_anomalies": ["..."],
            "ela_analysis": "...",
            "final_explanation": "..."
        }}
        """

        # Encode images for API
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        with open(ela_path, "rb") as f:
            ela_data = base64.b64encode(f.read()).decode("utf-8")

        # Multi-modal Message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ela_data}"}}
            ]
        )

        try:
            response = self.llm.invoke([message])
            # Parse Result
            analysis_json = json.loads(response.content.replace('```json', '').replace('```', ''))
        except Exception as e:
            print(f"❌ Deepfake Analysis Failed: {e}")
            analysis_json = {"error": str(e), "verdict": "UNKNOWN", "confidence_score": 0}

        return {
            "meta": meta_report,
            "ai_analysis": analysis_json,
            "ela_image_path": os.path.basename(ela_path) # Send filename back to UI
        }
