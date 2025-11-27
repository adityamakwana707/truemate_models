"""
Enhanced TruthMate ML Service with Real Models
Advanced fact-checking pipeline using state-of-the-art ML models
"""
import os
import json
import re
import time
import logging
import asyncio
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import nltk
from googlesearch import search
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'])

class EnhancedFactChecker:
    def __init__(self):
        """Initialize all ML models and components"""
        self.models_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.load_models()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Known fact-checking datasets for similarity matching
        self.fact_db = self.load_fact_database()
        
    def load_models(self):
        """Load all required ML models"""
        try:
            logger.info("Loading ML models...")
            
            # 1. Fake News Detection Model (DistilBERT fine-tuned)
            self.fake_news_tokenizer = AutoTokenizer.from_pretrained(
                "jy46604790/Fake-News-Bert-Detect"
            )
            self.fake_news_model = AutoModelForSequenceClassification.from_pretrained(
                "jy46604790/Fake-News-Bert-Detect"
            ).to(self.device)
            
            # 2. Stance Detection Model
            self.stance_pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-stance-detection",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 3. Bias Detection Model
            self.bias_pipeline = pipeline(
                "text-classification",
                model="d4data/bias-detection-model",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 4. Sentence Transformer for Semantic Similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 5. Credibility Classification
            self.credibility_pipeline = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    def load_fact_database(self) -> List[Dict]:
        """Load a database of known facts for similarity matching"""
        # This would typically load from a real fact-checking database
        # For now, we'll use a small sample
        return [
            {
                "claim": "COVID-19 vaccines are safe and effective",
                "verdict": "True",
                "sources": ["CDC", "WHO", "FDA"],
                "confidence": 0.95
            },
            {
                "claim": "The Earth is flat",
                "verdict": "False", 
                "sources": ["NASA", "Scientific Consensus"],
                "confidence": 0.99
            },
            {
                "claim": "Coffee can reduce risk of heart disease",
                "verdict": "True",
                "sources": ["American Heart Association", "Harvard Health"],
                "confidence": 0.82
            },
            {
                "claim": "5G causes coronavirus",
                "verdict": "False",
                "sources": ["WHO", "Scientific Studies"],
                "confidence": 0.97
            }
        ]
    
    def verify_claim(self, text: str) -> Dict:
        """Main claim verification using ensemble of models"""
        try:
            # 1. Fake news detection
            fake_score = self.detect_fake_news(text)
            
            # 2. Similarity matching with fact database
            similarity_result = self.check_similarity_database(text)
            
            # 3. Content analysis
            content_analysis = self.analyze_content_quality(text)
            
            # 4. Source credibility (if URLs found)
            urls = self.extract_urls(text)
            source_credibility = self.analyze_source_credibility(urls) if urls else {"avg_credibility": 0.5}
            
            # Ensemble decision
            verdict, confidence = self.make_ensemble_decision(
                fake_score, similarity_result, content_analysis, source_credibility
            )
            
            return {
                "label": verdict,
                "confidence": int(confidence * 100),
                "explanation": self.generate_explanation(verdict, confidence, text),
                "reasoning": self.generate_reasoning(fake_score, similarity_result, content_analysis),
                "evidence_queries": self.generate_evidence_queries(text),
                "source_analysis": source_credibility
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
    
    def detect_fake_news(self, text: str) -> Dict:
        """Use fine-tuned BERT model for fake news detection"""
        try:
            inputs = self.fake_news_tokenizer(
                text, return_tensors="pt", truncation=True, 
                padding=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.fake_news_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Assuming labels: 0=Real, 1=Fake
            fake_prob = predictions[0][1].item()
            real_prob = predictions[0][0].item()
            
            return {
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "predicted_class": "Fake" if fake_prob > real_prob else "Real"
            }
            
        except Exception as e:
            logger.error(f"Error in fake news detection: {e}")
            return {"fake_probability": 0.5, "real_probability": 0.5, "predicted_class": "Unknown"}
    
    def check_similarity_database(self, text: str) -> Dict:
        """Check similarity with known fact-checked claims"""
        try:
            text_embedding = self.sentence_model.encode([text])
            
            best_match = None
            highest_similarity = 0
            
            for fact in self.fact_db:
                fact_embedding = self.sentence_model.encode([fact["claim"]])
                similarity = cosine_similarity(text_embedding, fact_embedding)[0][0]
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = fact
            
            return {
                "best_match": best_match,
                "similarity_score": float(highest_similarity),
                "threshold_met": highest_similarity > 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in similarity check: {e}")
            return {"best_match": None, "similarity_score": 0, "threshold_met": False}
    
    def analyze_content_quality(self, text: str) -> Dict:
        """Analyze text quality indicators"""
        try:
            # Language quality metrics
            blob = TextBlob(text)
            
            # Count suspicious patterns
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
            exclamation_count = text.count('!')
            question_count = text.count('?')
            word_count = len(text.split())
            
            # Emotional language detection
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Quality score calculation
            quality_score = 1.0
            if caps_ratio > 0.3:  # Too many caps
                quality_score -= 0.2
            if exclamation_count > 3:  # Too many exclamations
                quality_score -= 0.1
            if word_count < 10:  # Too short
                quality_score -= 0.2
            if abs(sentiment['compound']) > 0.8:  # Very emotional
                quality_score -= 0.1
                
            return {
                "quality_score": max(0, quality_score),
                "caps_ratio": caps_ratio,
                "emotional_indicators": sentiment,
                "word_count": word_count,
                "suspicious_patterns": caps_ratio > 0.3 or exclamation_count > 3
            }
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return {"quality_score": 0.5, "suspicious_patterns": False}
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def analyze_source_credibility(self, urls: List[str]) -> Dict:
        """Analyze credibility of sources"""
        try:
            # Known credible domains
            credible_domains = {
                'bbc.com': 0.9, 'reuters.com': 0.9, 'apnews.com': 0.9,
                'cdc.gov': 0.95, 'who.int': 0.95, 'nature.com': 0.9,
                'sciencedirect.com': 0.85, 'pubmed.ncbi.nlm.nih.gov': 0.9,
                'harvard.edu': 0.85, 'stanford.edu': 0.85, 'mit.edu': 0.85
            }
            
            # Known unreliable domains  
            unreliable_domains = {
                'infowars.com': 0.1, 'beforeitsnews.com': 0.2,
                'naturalnews.com': 0.2, 'zerohedge.com': 0.3
            }
            
            credibility_scores = []
            analyzed_sources = []
            
            for url in urls[:5]:  # Limit to 5 URLs
                try:
                    domain = url.split('//')[1].split('/')[0].lower()
                    
                    if domain in credible_domains:
                        score = credible_domains[domain]
                    elif domain in unreliable_domains:
                        score = unreliable_domains[domain]
                    else:
                        # Default scoring based on domain characteristics
                        score = 0.5  # Neutral
                        if any(tld in domain for tld in ['.edu', '.gov', '.org']):
                            score += 0.2
                        if any(word in domain for word in ['news', 'times', 'post']):
                            score += 0.1
                    
                    credibility_scores.append(score)
                    analyzed_sources.append({"url": url, "domain": domain, "credibility": score})
                    
                except Exception:
                    continue
            
            avg_credibility = np.mean(credibility_scores) if credibility_scores else 0.5
            
            return {
                "avg_credibility": float(avg_credibility),
                "source_count": len(analyzed_sources),
                "sources": analyzed_sources,
                "has_credible_sources": avg_credibility > 0.7
            }
            
        except Exception as e:
            logger.error(f"Error in source analysis: {e}")
            return {"avg_credibility": 0.5, "source_count": 0, "sources": []}
    
    def make_ensemble_decision(self, fake_score: Dict, similarity_result: Dict, 
                             content_analysis: Dict, source_credibility: Dict) -> Tuple[str, float]:
        """Combine all signals to make final decision"""
        try:
            # Weight different signals
            weights = {
                'fake_news_model': 0.4,
                'similarity_match': 0.3,
                'content_quality': 0.2,
                'source_credibility': 0.1
            }
            
            # Calculate weighted score (higher = more likely true)
            score = 0.0
            
            # Fake news model contribution (inverted)
            fake_prob = fake_score.get('fake_probability', 0.5)
            score += weights['fake_news_model'] * (1 - fake_prob)
            
            # Similarity match contribution
            if similarity_result.get('threshold_met', False):
                match_verdict = similarity_result['best_match']['verdict']
                if match_verdict == 'True':
                    score += weights['similarity_match'] * similarity_result['similarity_score']
                elif match_verdict == 'False':
                    score += weights['similarity_match'] * (1 - similarity_result['similarity_score'])
                else:  # Misleading
                    score += weights['similarity_match'] * 0.3
            else:
                score += weights['similarity_match'] * 0.5  # Neutral if no match
            
            # Content quality contribution
            quality_score = content_analysis.get('quality_score', 0.5)
            score += weights['content_quality'] * quality_score
            
            # Source credibility contribution
            credibility = source_credibility.get('avg_credibility', 0.5)
            score += weights['source_credibility'] * credibility
            
            # Determine verdict and confidence
            if score > 0.7:
                verdict = "True"
                confidence = min(0.95, score)
            elif score < 0.3:
                verdict = "False"
                confidence = min(0.95, 1 - score)
            elif 0.3 <= score <= 0.7:
                if abs(score - 0.5) < 0.1:
                    verdict = "Unknown"
                    confidence = 0.6
                else:
                    verdict = "Misleading"
                    confidence = 0.7
            else:
                verdict = "Unknown"
                confidence = 0.5
                
            return verdict, confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble decision: {e}")
            return "Unknown", 0.5
    
    def generate_explanation(self, verdict: str, confidence: float, text: str) -> str:
        """Generate human-readable explanation"""
        explanations = {
            "True": f"Our analysis indicates this claim is likely true with {confidence*100:.1f}% confidence. The statement appears to be supported by reliable sources and aligns with factual information.",
            "False": f"Our analysis suggests this claim is likely false with {confidence*100:.1f}% confidence. The statement contradicts established facts or reliable sources.",
            "Misleading": f"This claim appears to be misleading with {confidence*100:.1f}% confidence. While it may contain some truthful elements, it lacks important context or contains inaccuracies.",
            "Unknown": f"We cannot determine the veracity of this claim with sufficient confidence. More evidence or context may be needed for accurate verification."
        }
        return explanations.get(verdict, "Unable to generate explanation.")
    
    def generate_reasoning(self, fake_score: Dict, similarity_result: Dict, content_analysis: Dict) -> str:
        """Generate detailed reasoning"""
        reasoning_parts = []
        
        # Fake news model reasoning
        fake_prob = fake_score.get('fake_probability', 0.5)
        if fake_prob > 0.7:
            reasoning_parts.append("Our fake news detection model flagged this content as potentially false.")
        elif fake_prob < 0.3:
            reasoning_parts.append("Our fake news detection model suggests this content is likely authentic.")
        
        # Similarity matching reasoning
        if similarity_result.get('threshold_met', False):
            match = similarity_result['best_match']
            reasoning_parts.append(f"This claim is similar to a previously fact-checked statement that was found to be {match['verdict']}.")
        
        # Content quality reasoning
        if content_analysis.get('suspicious_patterns', False):
            reasoning_parts.append("The text contains patterns often associated with misinformation (excessive capitalization, emotional language, etc.).")
        
        return " ".join(reasoning_parts) if reasoning_parts else "Analysis completed using multiple verification methods."
    
    def generate_evidence_queries(self, text: str) -> List[str]:
        """Generate search queries for evidence gathering"""
        # Extract key terms and generate search queries
        blob = TextBlob(text)
        noun_phrases = list(blob.noun_phrases)[:3]
        
        queries = [
            f"{text} fact check",
            f"{text} verification",
            f"{text} debunked"
        ]
        
        for phrase in noun_phrases:
            if len(phrase.split()) > 1:
                queries.append(f"{phrase} scientific study")
                queries.append(f"{phrase} reliable source")
        
        return queries[:5]

# Initialize the fact checker
fact_checker = EnhancedFactChecker()

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check"""
    return jsonify({
        "status": "ok",
        "models_loaded": fact_checker.models_loaded,
        "device": str(fact_checker.device),
        "timestamp": time.time(),
        "service": "TruthMate Enhanced ML Service",
        "version": "2.0.0"
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
        
        # Use enhanced fact checker
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

# Keep other endpoints with enhanced functionality...
@app.route('/stance-detection', methods=['POST'])
def stance_detection():
    """Enhanced stance detection"""
    try:
        data = request.get_json()
        claim = data.get('claim', '')
        
        if not claim:
            return jsonify({"error": "No claim provided"}), 400
        
        # Use the stance detection pipeline
        if fact_checker.models_loaded:
            try:
                result = fact_checker.stance_pipeline(claim)
                stance_label = result[0]['label']
                stance_confidence = result[0]['score']
                
                return jsonify({
                    "stance": stance_label.lower(),
                    "confidence": int(stance_confidence * 100),
                    "sources": fact_checker.generate_evidence_queries(claim)
                })
            except Exception as e:
                logger.error(f"Stance detection model error: {e}")
        
        # Fallback
        return jsonify({
            "stance": "neutral",
            "confidence": 50,
            "sources": []
        })
        
    except Exception as e:
        logger.error(f"Error in stance detection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/source-credibility', methods=['POST'])
def source_credibility():
    """Enhanced source credibility analysis"""
    try:
        data = request.get_json()
        queries = data.get('queries', [])
        
        if not queries:
            return jsonify({"error": "No queries provided"}), 400
        
        all_sources = []
        total_credibility = 0
        
        for query in queries[:3]:  # Limit to 3 queries
            # Extract URLs from query if any
            urls = fact_checker.extract_urls(query)
            if urls:
                credibility_result = fact_checker.analyze_source_credibility(urls)
                all_sources.extend(credibility_result.get('sources', []))
                total_credibility += credibility_result.get('avg_credibility', 0.5)
        
        avg_credibility = total_credibility / len(queries) if queries else 0.5
        
        return jsonify({
            "credible_sources": all_sources,
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
        
        # Sentiment analysis
        sentiment_scores = fact_checker.sentiment_analyzer.polarity_scores(text)
        
        # Bias detection (if model is loaded)
        bias_result = "neutral"
        bias_confidence = 50
        
        if fact_checker.models_loaded:
            try:
                bias_prediction = fact_checker.bias_pipeline(text)
                bias_result = bias_prediction[0]['label'].lower()
                bias_confidence = int(bias_prediction[0]['score'] * 100)
            except Exception as e:
                logger.error(f"Bias detection error: {e}")
        
        # Determine emotion based on sentiment
        compound = sentiment_scores['compound']
        if compound >= 0.5:
            emotion = "joy"
        elif compound <= -0.5:
            emotion = "sadness"
        elif sentiment_scores['neg'] > 0.3:
            emotion = "anger"
        else:
            emotion = "neutral"
        
        return jsonify({
            "bias": bias_result,
            "bias_confidence": bias_confidence,
            "sentiment": "positive" if compound > 0.1 else "negative" if compound < -0.1 else "neutral",
            "sentiment_confidence": int(abs(compound) * 100),
            "emotion": emotion,
            "emotion_confidence": int(max(sentiment_scores.values()) * 100)
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
        confidence = data.get('confidence', 0)
        
        if not claim:
            return jsonify({"error": "Claim required"}), 400
        
        # Generate enhanced explanation
        explanation = fact_checker.generate_explanation(verdict, confidence/100, claim)
        
        # Generate comprehensive reasoning
        fake_score = fact_checker.detect_fake_news(claim)
        similarity_result = fact_checker.check_similarity_database(claim)
        content_analysis = fact_checker.analyze_content_quality(claim)
        
        reasoning = fact_checker.generate_reasoning(fake_score, similarity_result, content_analysis)
        
        return jsonify({
            "explanation": explanation,
            "reasoning": reasoning,
            "evidence_queries": fact_checker.generate_evidence_queries(claim)
        })
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return jsonify({"error": str(e)}), 500

# Keep other endpoints as before (extract-claim, verify-image)
@app.route('/extract-claim', methods=['POST'])
def extract_claim():
    """Extract claims from URLs - enhanced version"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        
        # Simple URL content extraction
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title and first few paragraphs
            title = soup.find('title')
            title_text = title.text.strip() if title else ""
            
            paragraphs = soup.find_all('p')[:3]
            content = " ".join([p.text.strip() for p in paragraphs])
            
            extracted_text = f"{title_text} {content}"[:500]
            
            return jsonify({
                "extracted_claim": extracted_text,
                "url": url,
                "title": title_text
            })
            
        except Exception as e:
            return jsonify({
                "extracted_claim": f"Unable to extract content from {url}",
                "error": str(e)
            })
            
    except Exception as e:
        logger.error(f"Error in extract-claim: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/verify-image', methods=['POST'])
def verify_image():
    """Image verification placeholder - would need specialized models"""
    try:
        data = request.get_json()
        image = data.get('image', '')
        
        if not image:
            return jsonify({"error": "No image provided"}), 400
        
        # Placeholder for image verification
        # Would implement reverse image search, deepfake detection, etc.
        
        return jsonify({
            "label": "Unknown",
            "confidence": 50,
            "explanation": "Image verification requires specialized computer vision models. Currently not implemented.",
            "techniques": ["reverse_image_search", "deepfake_detection", "metadata_analysis"]
        })
        
    except Exception as e:
        logger.error(f"Error in verify-image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting TruthMate Enhanced ML Service...")
    print("📍 Running on http://localhost:5000") 
    print(f"🖥️  Using device: {fact_checker.device}")
    print(f"🤖 Models loaded: {fact_checker.models_loaded}")
    print("🔍 Enhanced endpoints available:")
    print("   GET  /health - Service health check")
    print("   POST /verify - Advanced claim verification") 
    print("   POST /stance-detection - Stance analysis")
    print("   POST /source-credibility - Source reliability")
    print("   POST /bias-sentiment - Bias & sentiment analysis")
    print("   POST /generate-explanation - Detailed explanations")
    print("   POST /extract-claim - URL content extraction")
    print("   POST /verify-image - Image verification (placeholder)")
    print("\n🔗 Connect your Next.js app to http://localhost:5000")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)