"""
TruthMate Mock ML Service
A development Flask server that implements all ML endpoints with mock responses
Run this to test your Next.js integration before deploying real models
"""
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import time
import random
import json

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])  # Enable CORS for Next.js requests

# Ensure proper JSON responses
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Mock API key for development
MOCK_API_KEY = "dev-api-key-123"

def verify_auth():
    """Simple API key verification for development"""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer ') and auth_header != f'Bearer {MOCK_API_KEY}':
        # Allow requests without auth in development
        pass
    return True

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    response_data = {
        "status": "ok",
        "model_loaded": True,
        "timestamp": time.time(),
        "service": "TruthMate Mock ML Service"
    }
    
    response = jsonify(response_data)
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/verify', methods=['POST'])
def verify_claim():
    """Main classification endpoint"""
    try:
        verify_auth()
        
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
            
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        print(f"Received verification request for text: {text[:50]}...")
    except Exception as e:
        print(f"Error in verify_claim: {e}")
        return jsonify({"error": "Invalid request format"}), 400
    
    # Mock classification logic
    verdicts = ['True', 'False', 'Misleading', 'Unknown']
    
    # Simple mock logic based on keywords
    text_lower = text.lower()
    if any(word in text_lower for word in ['true', 'fact', 'research', 'study']):
        verdict = 'True'
        confidence = random.randint(70, 95)
    elif any(word in text_lower for word in ['false', 'fake', 'hoax', 'conspiracy']):
        verdict = 'False'
        confidence = random.randint(75, 90)
    elif any(word in text_lower for word in ['misleading', 'partial', 'context']):
        verdict = 'Misleading'
        confidence = random.randint(60, 80)
    else:
        verdict = random.choice(verdicts)
        confidence = random.randint(40, 85)
    
    response_data = {
        "label": verdict,
        "confidence": confidence,
        "explanation": f"Based on analysis, this claim appears to be {verdict.lower()}.",
        "reasoning": f"The text contains indicators suggesting a {verdict.lower()} classification.",
        "evidence_queries": [f"{text} fact check", f"{text} verification", "reliable sources"]
    }
    
    response = jsonify(response_data)
    response.headers['Content-Type'] = 'application/json'
    print(f"Sending response: {response_data}")
    return response

@app.route('/stance-detection', methods=['POST'])
def stance_detection():
    """Stance detection endpoint"""
    verify_auth()
    
    data = request.get_json()
    claim = data.get('claim', '')
    evidence_queries = data.get('evidence_queries', [])
    
    # Mock stance analysis
    stances = ['supports', 'refutes', 'neutral', 'unknown']
    
    mock_sources = []
    for i, query in enumerate(evidence_queries[:3]):  # Limit to 3 sources
        mock_sources.append({
            "title": f"Analysis of: {query[:50]}...",
            "url": f"https://example-source-{i+1}.com/article",
            "snippet": f"This source provides information about {query[:30]}...",
            "credibility": random.uniform(0.6, 0.9),
            "relevance": random.uniform(0.7, 0.95),
            "stance": random.choice(stances)
        })
    
    return jsonify({
        "stance": random.choice(stances),
        "confidence": random.randint(60, 85),
        "sources": mock_sources
    })

@app.route('/source-credibility', methods=['POST'])
def source_credibility():
    """Source credibility analysis"""
    verify_auth()
    
    data = request.get_json()
    queries = data.get('queries', [])
    
    # Mock credibility scoring
    credible_sources = []
    for query in queries:
        credible_sources.append({
            "source": f"FactCheck-{random.randint(1,100)}.org",
            "credibility_score": random.uniform(0.7, 0.95),
            "bias_rating": random.choice(['left', 'center', 'right']),
            "factual_reporting": random.choice(['high', 'mixed', 'low'])
        })
    
    avg_credibility = sum(s['credibility_score'] for s in credible_sources) / len(credible_sources) if credible_sources else 0
    
    return jsonify({
        "credible_sources": credible_sources,
        "avg_credibility": avg_credibility * 100,  # Convert to percentage
        "confidence": random.randint(70, 90)
    })

@app.route('/extract-claim', methods=['POST'])
def extract_claim():
    """Claim extraction from URL or text"""
    verify_auth()
    
    data = request.get_json()
    url = data.get('url', '')
    text = data.get('text', '')
    
    if url:
        # Mock URL processing
        extracted = f"Main claim extracted from {url}: Sample claim about the topic"
    elif text:
        # Mock text processing - extract first sentence
        sentences = text.split('.')
        extracted = sentences[0] if sentences else text
    else:
        return jsonify({"error": "No URL or text provided"}), 400
    
    return jsonify({
        "extracted_claim": extracted,
        "confidence": random.randint(75, 90),
        "explanation": extracted
    })

@app.route('/verify-image', methods=['POST'])
def verify_image():
    """Image authenticity verification"""
    verify_auth()
    
    data = request.get_json()
    image = data.get('image', '')
    image_url = data.get('image_url', '')
    
    if not image and not image_url:
        return jsonify({"error": "No image provided"}), 400
    
    # Mock image analysis
    authenticity_options = ['real', 'fake', 'ai-generated', 'unknown']
    authenticity = random.choice(authenticity_options)
    confidence = random.randint(60, 85)
    
    return jsonify({
        "authenticity": authenticity,
        "confidence": confidence,
        "analysis": {
            "deepfake_probability": random.uniform(0.1, 0.8),
            "manipulation_detected": random.choice([True, False]),
            "ai_generated_probability": random.uniform(0.2, 0.7)
        }
    })

@app.route('/bias-sentiment', methods=['POST'])
def bias_sentiment():
    """Bias and sentiment analysis"""
    verify_auth()
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Mock bias and sentiment analysis
    bias_options = ['left', 'center', 'right', 'neutral']
    sentiment_options = ['positive', 'negative', 'neutral']
    emotion_options = ['neutral', 'anger', 'fear', 'joy', 'sadness']
    
    return jsonify({
        "bias": random.choice(bias_options),
        "bias_confidence": random.randint(60, 85),
        "sentiment": random.choice(sentiment_options),
        "sentiment_confidence": random.randint(65, 90),
        "emotion": random.choice(emotion_options),
        "emotion_confidence": random.randint(55, 80)
    })

@app.route('/generate-explanation', methods=['POST'])
def generate_explanation():
    """Generate comprehensive explanation"""
    verify_auth()
    
    data = request.get_json()
    claim = data.get('claim', '')
    verdict = data.get('verdict', 'Unknown')
    confidence = data.get('confidence', 0)
    
    if not claim or not verdict:
        return jsonify({"error": "Claim and verdict required"}), 400
    
    # Generate mock explanation
    explanation = f"After analyzing the claim '{claim}', our AI models determined it is {verdict} with {confidence}% confidence."
    
    reasoning = f"""Our analysis considered multiple factors:
    1. Content analysis of the claim text
    2. Cross-referencing with reliable sources
    3. Evaluating source credibility and bias
    4. Assessing emotional language and potential manipulation
    
    Based on these factors, we classified this claim as {verdict}."""
    
    return jsonify({
        "explanation": explanation,
        "reasoning": reasoning
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting TruthMate Mock ML Service...")
    print("📍 Running on http://localhost:5000")
    print("🔍 Available endpoints:")
    print("   GET  /health")
    print("   POST /verify")
    print("   POST /stance-detection")
    print("   POST /source-credibility")
    print("   POST /extract-claim")
    print("   POST /verify-image")
    print("   POST /bias-sentiment")
    print("   POST /generate-explanation")
    print("\n🔗 Connect your Next.js app to http://localhost:5000")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)