from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from agents.link_agent import LinkSafetyAgent
from agents.research import ResearchAgent
from agents.linguistics_forensics import LinguisticAgent, ForensicsAgent
from agents.deepfake_agent import DeepfakeAgent
import uvicorn
import os

app = FastAPI(title="TruthMate Backend")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Agents
link_agent = LinkSafetyAgent()
research_agent = ResearchAgent()
linguist_agent = LinguisticAgent()
forensics_agent = ForensicsAgent()
deepfake_agent = DeepfakeAgent()

# Mount static directory for ELA images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the single-file UI"""
    # Fix: Resolve index.html relative to this script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return f"<h1>Error: index.html not found at {index_path}</h1>"

@app.post("/api/link-check")
async def link_check(url: str = Form(...)):
    """Route for Link Safety Agent - Starts Session"""
    return await link_agent.start_session(url)

@app.post("/api/interact/click")
async def interact_click(x: int = Form(...), y: int = Form(...)):
    """Route for Remote Click Interaction"""
    return await link_agent.click_at(x, y)

@app.post("/api/interact/login")
async def interact_login(username: str = Form(...), password: str = Form(...)):
    """Route for Automated Login Attempt"""
    return await link_agent.attempt_login(username, password)

@app.post("/api/research")
async def research(claim: str = Form(...)):
    """Route for Research/Graph Agent"""
    return research_agent.build_consensus_graph(claim)

@app.post("/api/text-analysis")
async def text_analysis(text: str = Form(...)):
    """Route for Linguistics Agent"""
    ai_check = linguist_agent.detect_ai_text(text)
    propaganda = linguist_agent.analyze_propaganda(text)
    return {"ai_detection": ai_check, "propaganda_analysis": propaganda}

@app.post("/api/media-check")
async def media_check(file: UploadFile = File(...)):
    """Route for Media Forensics (Simulated for Demo)"""
    return {"message": "Media analysis is currently simulated for safety.", "status": "Files received"}

@app.post("/api/deepfake-check")
async def deepfake_check(file: UploadFile = File(...)):
    """Route for Deepfake Detection Agent"""
    # Save uploaded file
    file_location = f"static/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # Run Analysis
    return deepfake_agent.analyze(file_location)

if __name__ == "__main__":
    # Suppress GRPC logs
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"
    
    print("\n" + "="*50)
    print("🚀 TruthMate OS Backend is STARTING...")
    print("👉 Open your browser at: http://localhost:8000")
    print("="*50 + "\n")
    
    # Ensure browsers are installed: playwright install chromium
    uvicorn.run(app, host="0.0.0.0", port=8000)
