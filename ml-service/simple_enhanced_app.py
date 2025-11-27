"""
Simplified Enhanced TruthMate ML Service
A lightweight version with core improvements using minimal dependencies
"""
import os
import json
import re
import time
import random
import logging
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])

class SimplifiedFactChecker:
    def __init__(self):
        """Initialize the simplified fact checker with rule-based logic"""
        self.fact_database = self.load_fact_database()
        logger.info("Simplified fact checker initialized")
    
    def load_fact_database(self) -> List[Dict]:
        """Load a database of known facts for similarity matching"""
        return [
            {
                "keywords": ["covid", "vaccine", "safe", "effective"],
                "verdict": "True",
                "confidence": 0.90,
                "explanation": "COVID-19 vaccines have been proven safe and effective through extensive clinical trials."
            },
            {
                "keywords": ["earth", "flat"],
                "verdict": "False",
                "confidence": 0.99,
                "explanation": "The Earth is scientifically proven to be spherical, not flat."
            },
            {
                "keywords": ["coffee", "heart", "disease", "health"],
                "verdict": "True", 
                "confidence": 0.75,
                "explanation": "Moderate coffee consumption has been associated with reduced risk of heart disease in studies."
            },
            {
                "keywords": ["5g", "coronavirus", "covid", "cause"],
                "verdict": "False",
                "confidence": 0.95,
                "explanation": "There is no scientific evidence linking 5G technology to coronavirus transmission."
            },
            {
                "keywords": ["climate", "change", "hoax", "fake"],
                "verdict": "False",
                "confidence": 0.92,
                "explanation": "Climate change is real and supported by overwhelming scientific consensus."
            },
            {
                "keywords": ["vaccine", "microchip", "tracking", "bill", "gates"],
                "verdict": "False",
                "confidence": 0.96,
                "explanation": "Vaccines do not contain microchips or tracking devices. This is a debunked conspiracy theory."
            },
            {
                "keywords": ["exercise", "health", "cardiovascular", "heart"],
                "verdict": "True",
                "confidence": 0.88,
                "explanation": "Regular exercise is scientifically proven to improve cardiovascular health."
            },
            {
                "keywords": ["smoking", "cancer", "lung", "risk"],
                "verdict": "True",
                "confidence": 0.95,
                "explanation": "Smoking significantly increases the risk of lung cancer and other diseases."
            }
        ]
    
    def analyze_text_quality(self, text: str) -> Dict:
        """Analyze text quality indicators"""
        # Count suspicious patterns
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        exclamation_count = text.count('!')
        question_count = text.count('?')
        word_count = len(text.split())
        
        # Check for emotional language
        emotional_words = ['outrageous', 'shocking', 'amazing', 'incredible', 'unbelievable', 
                          'terrible', 'awful', 'disgusting', 'fantastic', 'brilliant']
        emotional_count = sum(1 for word in emotional_words if word.lower() in text.lower())
        
        # Quality score calculation
        quality_score = 1.0
        if caps_ratio > 0.3:  # Too many caps
            quality_score -= 0.2
        if exclamation_count > 3:  # Too many exclamations
            quality_score -= 0.1
        if word_count < 5:  # Too short
            quality_score -= 0.3
        if emotional_count > 2:  # Very emotional language
            quality_score -= 0.15
        
        return {
            "quality_score": max(0, quality_score),
            "caps_ratio": caps_ratio,
            "exclamation_count": exclamation_count,
            "word_count": word_count,
            "emotional_words": emotional_count,
            "suspicious_patterns": caps_ratio > 0.3 or exclamation_count > 3 or emotional_count > 2
        }
    
    def check_fact_database(self, text: str) -> Dict:
        """Check against known fact database using keyword matching"""
        text_lower = text.lower()
        best_match = None
        highest_score = 0
        
        for fact in self.fact_database:
            # Count keyword matches
            matches = sum(1 for keyword in fact["keywords"] if keyword in text_lower)
            score = matches / len(fact["keywords"])
            
            if score > highest_score and matches >= 2:  # Need at least 2 keyword matches
                highest_score = score
                best_match = fact
        
        return {
            "match_found": best_match is not None,
            "match_score": highest_score,
            "matched_fact": best_match
        }
    
    def analyze_sentiment_simple(self, text: str) -> Dict:
        """Simple sentiment analysis using keyword lists"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'beneficial', 'helpful', 'positive', 'effective', 'proven', 'scientific']
        negative_words = ['bad', 'terrible', 'awful', 'dangerous', 'harmful', 'fake', 'false', 
                         'hoax', 'conspiracy', 'lie', 'scam', 'fraud', 'misleading']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(80, 50 + positive_count * 10)
        elif negative_count > positive_count:
            sentiment = "negative" 
            confidence = min(80, 50 + negative_count * 10)
        else:
            sentiment = "neutral"
            confidence = 50
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def detect_bias_simple(self, text: str) -> Dict:
        """Simple bias detection using keyword patterns"""
        left_indicators = ['progressive', 'liberal', 'social justice', 'equality', 'diversity']
        right_indicators = ['conservative', 'traditional', 'patriotic', 'law and order', 'security']
        neutral_indicators = ['research', 'study', 'data', 'evidence', 'scientific', 'peer-reviewed']
        
        text_lower = text.lower()
        left_count = sum(1 for indicator in left_indicators if indicator in text_lower)
        right_count = sum(1 for indicator in right_indicators if indicator in text_lower)
        neutral_count = sum(1 for indicator in neutral_indicators if indicator in text_lower)
        
        if neutral_count > max(left_count, right_count):
            bias = "neutral"
            confidence = min(85, 60 + neutral_count * 8)
        elif left_count > right_count:
            bias = "left"
            confidence = min(75, 50 + left_count * 10)
        elif right_count > left_count:
            bias = "right"
            confidence = min(75, 50 + right_count * 10)
        else:
            bias = "center"
            confidence = 60
        
        return {
            "bias": bias,
            "confidence": confidence,
            "left_indicators": left_count,
            "right_indicators": right_count,
            "neutral_indicators": neutral_count
        }
    
    def verify_claim(self, text: str) -> Dict:
        """Enhanced claim verification using rule-based analysis"""
        try:
            # 1. Check against fact database
            fact_match = self.check_fact_database(text)
            
            # 2. Analyze text quality
            quality_analysis = self.analyze_text_quality(text)
            
            # 3. Simple sentiment analysis
            sentiment_analysis = self.analyze_sentiment_simple(text)
            
            # 4. Bias detection
            bias_analysis = self.detect_bias_simple(text)
            
            # 5. Make decision using weighted scoring
            verdict, confidence = self.make_decision(fact_match, quality_analysis, sentiment_analysis)
            
            return {
                "label": verdict,
                "confidence": int(confidence * 100),
                "explanation": self.generate_explanation(verdict, confidence, fact_match, quality_analysis),
                "reasoning": self.generate_reasoning(fact_match, quality_analysis, sentiment_analysis, bias_analysis),
                "evidence_queries": self.generate_evidence_queries(text),
                "analysis": {
                    "fact_match": fact_match,
                    "quality": quality_analysis,
                    "sentiment": sentiment_analysis,
                    "bias": bias_analysis
                }
            }
            
        except Exception as e:
            logger.error(f"Error in verify_claim: {e}")
            return {
                "label": "Unknown",
                "confidence": 0,
                "explanation": f"Error analyzing claim: {str(e)}",
                "reasoning": "Technical error occurred during analysis",
                "evidence_queries": []
            }
    
    def make_decision(self, fact_match: Dict, quality_analysis: Dict, sentiment_analysis: Dict) -> Tuple[str, float]:
        """Make verdict decision using weighted scoring"""
        # Start with neutral score
        score = 0.5
        confidence = 0.6
        
        # Factor 1: Fact database match (40% weight)
        if fact_match["match_found"]:
            matched_fact = fact_match["matched_fact"]
            match_weight = fact_match["match_score"] * 0.4
            
            if matched_fact["verdict"] == "True":
                score += match_weight
            elif matched_fact["verdict"] == "False":
                score -= match_weight
            else:  # Misleading
                score += match_weight * 0.3
            
            confidence = max(confidence, matched_fact["confidence"])
        
        # Factor 2: Text quality (30% weight)
        quality_weight = quality_analysis["quality_score"] * 0.3
        score += (quality_weight - 0.15)  # Center around 0
        
        # Factor 3: Sentiment analysis (20% weight) 
        if sentiment_analysis["sentiment"] == "negative" and quality_analysis["suspicious_patterns"]:
            score -= 0.1  # Negative + suspicious = likely false
        elif sentiment_analysis["sentiment"] == "positive" and not quality_analysis["suspicious_patterns"]:
            score += 0.05  # Positive + quality = slight boost
        
        # Factor 4: Suspicious patterns (10% weight)
        if quality_analysis["suspicious_patterns"]:
            score -= 0.1
            confidence *= 0.9  # Reduce confidence for suspicious content
        
        # Convert score to verdict
        if score > 0.65:
            verdict = "True"
            final_confidence = min(0.9, confidence * (0.5 + score))
        elif score < 0.35:
            verdict = "False"  
            final_confidence = min(0.9, confidence * (1.5 - score))
        elif 0.45 <= score <= 0.55:
            verdict = "Unknown"
            final_confidence = 0.6
        else:
            verdict = "Misleading"
            final_confidence = min(0.8, confidence * 0.8)
        
        return verdict, final_confidence
    
    def generate_explanation(self, verdict: str, confidence: float, fact_match: Dict, quality_analysis: Dict) -> str:
        """Generate human-readable explanation"""
        base_explanations = {
            "True": f"This claim appears to be true with {confidence*100:.0f}% confidence.",
            "False": f"This claim appears to be false with {confidence*100:.0f}% confidence.",
            "Misleading": f"This claim appears to be misleading with {confidence*100:.0f}% confidence.",
            "Unknown": "We cannot determine the accuracy of this claim with sufficient confidence."
        }
        
        explanation = base_explanations.get(verdict, "Unable to analyze this claim.")
        
        # Add context from fact matching
        if fact_match["match_found"]:
            matched_explanation = fact_match["matched_fact"]["explanation"]
            explanation += f" {matched_explanation}"
        
        # Add quality context
        if quality_analysis["suspicious_patterns"]:
            explanation += " The text contains patterns often associated with unreliable information."
        
        return explanation
    
    def generate_reasoning(self, fact_match: Dict, quality_analysis: Dict, sentiment_analysis: Dict, bias_analysis: Dict) -> str:
        """Generate detailed reasoning"""
        reasoning_parts = []
        
        if fact_match["match_found"]:
            score = fact_match["match_score"]
            reasoning_parts.append(f"Found {score:.0%} similarity with known fact-checked content.")
        
        if quality_analysis["suspicious_patterns"]:
            patterns = []
            if quality_analysis["caps_ratio"] > 0.3:
                patterns.append("excessive capitalization")
            if quality_analysis["exclamation_count"] > 3:
                patterns.append("multiple exclamation marks")
            if quality_analysis["emotional_words"] > 2:
                patterns.append("emotional language")
            
            reasoning_parts.append(f"Text shows suspicious patterns: {', '.join(patterns)}.")
        
        if sentiment_analysis["sentiment"] != "neutral":
            reasoning_parts.append(f"Sentiment analysis indicates {sentiment_analysis['sentiment']} tone.")
        
        if bias_analysis["bias"] != "center":
            reasoning_parts.append(f"Content shows {bias_analysis['bias']} bias indicators.")
        
        return " ".join(reasoning_parts) if reasoning_parts else "Analysis based on content patterns and known facts."
    
    def generate_evidence_queries(self, text: str) -> List[str]:
        """Generate search queries for evidence gathering"""
        # Extract key terms (simple noun extraction)
        words = text.split()
        key_words = [word.strip('.,!?').lower() for word in words if len(word) > 3 and word.isalpha()]
        
        queries = [
            f"{' '.join(key_words[:3])} fact check",
            f"{' '.join(key_words[:2])} verification", 
            f"{text[:50]} reliable sources",
            f"{' '.join(key_words[:2])} scientific study",
            f"{text[:30]} debunked"
        ]
        
        return queries[:5]

# Initialize the enhanced fact checker
fact_checker = SimplifiedFactChecker()

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check"""
    return jsonify({
        "status": "ok",
        "service": "TruthMate Simplified Enhanced ML Service",
        "version": "1.5.0",
        "models_loaded": True,
        "capabilities": [
            "fact_database_matching",
            "text_quality_analysis", 
            "sentiment_analysis",
            "bias_detection",
            "evidence_query_generation"
        ],
        "timestamp": time.time()
    })

@app.route('/verify', methods=['POST'])
def verify_claim():
    """Enhanced claim verification endpoint"""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
            
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        logger.info(f"Verifying claim: {text[:100]}...")
        
        # Use simplified enhanced fact checker
        result = fact_checker.verify_claim(text)
        
        logger.info(f"Verification result: {result['label']} ({result['confidence']}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in verify endpoint: {e}")
        return jsonify({
            "error": f"Verification failed: {str(e)}",
            "label": "Unknown",
            "confidence": 0
        }), 500

@app.route('/stance-detection', methods=['POST'])
def stance_detection():
    """Simplified stance detection"""
    try:
        data = request.get_json()
        claim = data.get('claim', '')
        
        if not claim:
            return jsonify({"error": "No claim provided"}), 400
        
        # Simple stance detection based on keywords
        support_keywords = ['support', 'agree', 'endorse', 'favor', 'back', 'yes', 'true', 'correct']
        oppose_keywords = ['oppose', 'disagree', 'against', 'reject', 'no', 'false', 'wrong', 'deny']
        
        claim_lower = claim.lower()
        support_count = sum(1 for keyword in support_keywords if keyword in claim_lower)
        oppose_count = sum(1 for keyword in oppose_keywords if keyword in claim_lower)
        
        if support_count > oppose_count:
            stance = "support"
            confidence = min(80, 50 + support_count * 10)
        elif oppose_count > support_count:
            stance = "oppose"
            confidence = min(80, 50 + oppose_count * 10)
        else:
            stance = "neutral"
            confidence = 50
        
        return jsonify({
            "stance": stance,
            "confidence": confidence,
            "sources": fact_checker.generate_evidence_queries(claim)
        })
        
    except Exception as e:
        logger.error(f"Error in stance detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/source-credibility', methods=['POST'])
def source_credibility():
    """Simplified source credibility analysis"""
    try:
        data = request.get_json()
        queries = data.get('queries', [])
        
        if not queries:
            return jsonify({"error": "No queries provided"}), 400
        
        # Simple domain credibility scoring
        credible_domains = {
            'bbc.com': 0.9, 'reuters.com': 0.9, 'apnews.com': 0.9,
            'cdc.gov': 0.95, 'who.int': 0.95, 'nature.com': 0.9,
            'harvard.edu': 0.85, 'stanford.edu': 0.85, 'mit.edu': 0.85,
            'wikipedia.org': 0.7, 'britannica.com': 0.8
        }
        
        sources = []
        total_credibility = 0
        
        for query in queries[:5]:
            # Extract domain if URL-like
            if 'http' in query:
                try:
                    domain = query.split('//')[1].split('/')[0].lower()
                    credibility = credible_domains.get(domain, 0.5)
                    sources.append({"url": query, "domain": domain, "credibility": credibility})
                    total_credibility += credibility
                except:
                    sources.append({"url": query, "domain": "unknown", "credibility": 0.5})
                    total_credibility += 0.5
            else:
                # For text queries, assign moderate credibility
                sources.append({"query": query, "credibility": 0.6})
                total_credibility += 0.6
        
        avg_credibility = total_credibility / len(queries) if queries else 0.5
        
        return jsonify({
            "credible_sources": sources,
            "avg_credibility": avg_credibility,
            "credibility_score": int(avg_credibility * 100)
        })
        
    except Exception as e:
        logger.error(f"Error in source credibility: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/bias-sentiment', methods=['POST'])
def bias_sentiment():
    """Enhanced bias and sentiment analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Use our enhanced analysis
        sentiment_result = fact_checker.analyze_sentiment_simple(text)
        bias_result = fact_checker.detect_bias_simple(text)
        
        # Determine emotion based on sentiment and content
        text_lower = text.lower()
        if any(word in text_lower for word in ['angry', 'outraged', 'furious', 'disgusted']):
            emotion = "anger"
            emotion_confidence = 75
        elif any(word in text_lower for word in ['scared', 'afraid', 'terrified', 'worried']):
            emotion = "fear"
            emotion_confidence = 70
        elif sentiment_result["sentiment"] == "positive":
            emotion = "joy"
            emotion_confidence = sentiment_result["confidence"]
        elif sentiment_result["sentiment"] == "negative":
            emotion = "sadness"
            emotion_confidence = sentiment_result["confidence"]
        else:
            emotion = "neutral"
            emotion_confidence = 60
        
        return jsonify({
            "bias": bias_result["bias"],
            "bias_confidence": bias_result["confidence"],
            "sentiment": sentiment_result["sentiment"],
            "sentiment_confidence": sentiment_result["confidence"],
            "emotion": emotion,
            "emotion_confidence": emotion_confidence
        })
        
    except Exception as e:
        logger.error(f"Error in bias-sentiment analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-explanation', methods=['POST'])
def generate_explanation():
    """Enhanced explanation generation"""
    try:
        data = request.get_json()
        claim = data.get('claim', '')
        verdict = data.get('verdict', 'Unknown')
        
        if not claim:
            return jsonify({"error": "Claim required"}), 400
        
        # Generate comprehensive analysis
        analysis_result = fact_checker.verify_claim(claim)
        
        explanation = analysis_result.get('explanation', f"This claim appears to be {verdict.lower()}.")
        reasoning = analysis_result.get('reasoning', 'Analysis based on available information.')
        
        return jsonify({
            "explanation": explanation,
            "reasoning": reasoning,
            "evidence_queries": analysis_result.get('evidence_queries', [])
        })
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return jsonify({"error": str(e)}), 500

# Keep existing endpoints for compatibility
@app.route('/extract-claim', methods=['POST'])
def extract_claim():
    """Simple claim extraction"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        
        # For now, return the URL as the claim (would need web scraping for full implementation)
        return jsonify({
            "extracted_claim": f"Content from {url}",
            "url": url,
            "title": "URL Content"
        })
        
    except Exception as e:
        logger.error(f"Error in extract-claim: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/verify-image', methods=['POST'])
def verify_image():
    """Image verification placeholder"""
    try:
        data = request.get_json()
        image = data.get('image', '')
        
        if not image:
            return jsonify({"error": "No image provided"}), 400
        
        return jsonify({
            "label": "Unknown",
            "confidence": 50,
            "explanation": "Image verification requires computer vision models. Feature coming soon.",
            "techniques": ["reverse_image_search", "metadata_analysis"]
        })
        
    except Exception as e:
        logger.error(f"Error in verify-image: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting TruthMate Simplified Enhanced ML Service...")
    print("📍 Running on http://localhost:5000")
    print("🧠 Using rule-based enhanced algorithms")
    print("🔍 Available endpoints:")
    print("   GET  /health - Service health check")
    print("   POST /verify - Enhanced claim verification")
    print("   POST /stance-detection - Stance analysis")
    print("   POST /source-credibility - Source reliability")
    print("   POST /bias-sentiment - Bias & sentiment analysis") 
    print("   POST /generate-explanation - Detailed explanations")
    print("   POST /extract-claim - URL content extraction")
    print("   POST /verify-image - Image verification (placeholder)")
    print("\n✨ Features:")
    print("   • Fact database matching")
    print("   • Text quality analysis")
    print("   • Sentiment & bias detection")
    print("   • Evidence query generation")
    print("   • Enhanced reasoning")
    print("\n🔗 Connect your Next.js app to http://localhost:5000")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)