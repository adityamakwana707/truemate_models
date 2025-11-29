import asyncio
import base64
import os
from playwright.async_api import async_playwright
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
import json

import requests

class LinkSafetyAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.MODEL_FAST, temperature=0)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.quarantine_dir = os.path.join(os.getcwd(), "truthmate_os", "quarantine_zone")
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def check_urlhaus(self, url: str):
        """
        Checks URL against URLHaus database for known malware.
        """
        try:
            # URLHaus API expects form data 'url'
            response = requests.post("https://urlhaus-api.abuse.ch/v1/url/", data={"url": url}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("query_status") == "ok" and data.get("url_status") == "online":
                    return {
                        "is_malware": True,
                        "tags": data.get("tags", []),
                        "threat": data.get("threat", "malware_download")
                    }
            return {"is_malware": False}
        except Exception as e:
            print(f"⚠️ URLHaus Check Failed: {e}")
            return {"is_malware": False, "error": str(e)}

    async def start_session(self, url: str):
        """
        Initializes a persistent browser session and navigates to the URL.
        """
        print(f"🕵️ Link Agent: Starting Interactive Session for {url}")
        
        # 1. Threat Intelligence Check
        threat_intel = self.check_urlhaus(url)
        if threat_intel["is_malware"]:
            print(f"🚨 CRITICAL WARNING: URLHaus flagged this site! Tags: {threat_intel.get('tags')}")

        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                user_agent=settings.USER_AGENT,
                viewport={'width': 1280, 'height': 800},
                accept_downloads=True
            )
            self.page = await self.context.new_page()
            
            # Event Listeners
            self.page.on("download", self._handle_download)

        try:
            response = await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
            report = await self._generate_report(url, response.status)
            
            # Merge Threat Intel into Report
            if threat_intel["is_malware"]:
                report["safety_verdict"]["risk_type"] = "MALWARE (Confirmed by URLHaus)"
                report["safety_verdict"]["safety_score"] = 0
                report["safety_verdict"]["reason"] = f"Flagged by URLHaus. Tags: {threat_intel.get('tags')}"
                
            return report
        except Exception as e:
            return {"error": str(e), "status": "error"}

    async def click_at(self, x: int, y: int):
        """
        Performs a click at the specified coordinates and returns an updated report.
        """
        if not self.page:
            return {"error": "No active session"}
        
        try:
            await self.page.mouse.click(x, y)
            await self.page.wait_for_timeout(1000) # Wait for UI update
            return await self._generate_report(self.page.url, 200)
        except Exception as e:
            return {"error": str(e)}

    async def attempt_login(self, username, password):
        """
        Heuristically attempts to find login fields and submit credentials.
        """
        if not self.page:
            return {"error": "No active session"}

        try:
            # Simple heuristic: look for input fields with specific types or names
            await self.page.fill('input[type="email"], input[name="email"], input[name="username"]', username)
            await self.page.fill('input[type="password"], input[name="password"]', password)
            
            # Try to find a submit button
            submit_selectors = ['button[type="submit"]', 'input[type="submit"]', 'button:has-text("Login")', 'button:has-text("Sign In")']
            for selector in submit_selectors:
                if await self.page.is_visible(selector):
                    await self.page.click(selector)
                    break
            
            await self.page.wait_for_timeout(2000)
            return await self._generate_report(self.page.url, 200)
        except Exception as e:
            return {"error": f"Login failed: {str(e)}"}

    async def _handle_download(self, download):
        """
        Intercepts downloads and saves them to the quarantine zone.
        """
        try:
            filename = download.suggested_filename
            filepath = os.path.join(self.quarantine_dir, filename)
            await download.save_as(filepath)
            
            heuristic_result = self._heuristic_check(filepath)
            print(f"⚠️ Download Intercepted: {filename} | Verdict: {heuristic_result}")
            # In a real app, we'd push this notification to the frontend
        except Exception as e:
            print(f"Download error: {e}")

    def _heuristic_check(self, filepath):
        """
        Checks file magic numbers to detect executables disguised as other files.
        """
        try:
            with open(filepath, "rb") as f:
                header = f.read(4)
            
            # PE Header (Windows Executable)
            if header.startswith(b'MZ'):
                return "SUSPICIOUS: Executable detected (MZ header)"
            
            # ELF Header (Linux Executable)
            if header.startswith(b'\x7fELF'):
                return "SUSPICIOUS: ELF Executable detected"
                
            return "SAFE: No executable signature detected"
        except Exception:
            return "UNKNOWN: Could not read file"

    async def _generate_report(self, url, status):
        """
        Generates the standard safety report with screenshot and AI analysis.
        """
        screenshot_bytes = await self.page.screenshot(full_page=False)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
        
        page_title = await self.page.title()
        content = await self.page.inner_text("body")
        content_snippet = content[:5000]

        prompt = f"""
        Analyze this website content.
        URL: {url}
        Title: {page_title}
        Content Snippet: {content_snippet}

        Task:
        1. Summarize what this link is about.
        2. Detect phishing, scams, or malware distribution language.
        3. Rate safety 0-100 (100 = Safe).
        
        Return JSON ONLY: {{ "summary": "...", "risk_type": "phishing|malware|safe|scam", "safety_score": int, "reason": "..." }}
        """
        
        try:
            ai_response = await self.llm.ainvoke(prompt)
            clean_json = ai_response.content.replace('```json', '').replace('```', '')
            analysis = json.loads(clean_json)
        except Exception as e:
            print(f"❌ AI Analysis Failed: {e}")
            analysis = {"summary": f"AI Analysis Failed: {str(e)}", "safety_score": 0, "risk_type": "unknown"}

        return {
            "url": url,
            "status": status,
            "screenshot_b64": screenshot_b64,
            "safety_verdict": analysis,
            "summary": analysis.get("summary", "No summary provided.")
        }

    async def close(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
