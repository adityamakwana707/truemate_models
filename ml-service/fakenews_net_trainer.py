"""
FakeNewsNet Dataset Training Pipeline for TruthMate
Advanced Training System with State-of-the-Art Models
"""
import os
import pandas as pd
import numpy as np
import requests
import zipfile
import json
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import shutil

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Advanced ML
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import joblib

# Data Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fakenews_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsNetTrainer:
    """Advanced FakeNewsNet dataset trainer for TruthMate platform"""
    
    def __init__(self, data_dir: str = "fakenews_data", models_dir: str = "trained_models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.setup_nltk()
        self.setup_spacy()
        
        # Dataset info
        self.fakenews_urls = {
            'politifact_fake': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_fake.csv',
            'politifact_real': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/politifact_real.csv',
            'gossipcop_fake': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_fake.csv',
            'gossipcop_real': 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/gossipcop_real.csv'
        }
        
        # Model configurations
        self.model_configs = {
            'advanced_tfidf': {'max_features': 50000, 'ngram_range': (1, 3), 'min_df': 2, 'max_df': 0.95},
            'enhanced_count': {'max_features': 30000, 'ngram_range': (1, 2), 'min_df': 3, 'max_df': 0.9},
            'char_ngrams': {'analyzer': 'char', 'ngram_range': (3, 5), 'max_features': 20000}
        }
        
        logger.info("FakeNewsNet Trainer initialized successfully")

    def setup_nltk(self):
        """Setup NLTK dependencies"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK setup completed")
        except Exception as e:
            logger.error(f"NLTK setup failed: {e}")

    def setup_spacy(self):
        """Setup spaCy NLP pipeline"""
        try:
            # Try to load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            logger.info("SpaCy setup completed")
        except Exception as e:
            logger.error(f"SpaCy setup failed: {e}")
            self.nlp = None

    def download_fakenews_data(self) -> bool:
        """Download FakeNewsNet CSV files"""
        logger.info("🔄 Downloading FakeNewsNet dataset...")
        
        downloaded_files = []
        for name, url in self.fakenews_urls.items():
            try:
                file_path = self.data_dir / f"{name}.csv"
                
                if file_path.exists():
                    logger.info(f"✅ {name}.csv already exists")
                    downloaded_files.append(file_path)
                    continue
                
                logger.info(f"📥 Downloading {name}...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_files.append(file_path)
                logger.info(f"✅ Downloaded {name}.csv ({len(response.content)} bytes)")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {name}: {e}")
                return False
        
        logger.info(f"🎉 Successfully downloaded {len(downloaded_files)} files")
        return True

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare FakeNewsNet data"""
        logger.info("📊 Loading and preparing FakeNewsNet data...")
        
        all_data = []
        
        for name, _ in self.fakenews_urls.items():
            file_path = self.data_dir / f"{name}.csv"
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
                logger.info(f"📋 Loaded {name}: {len(df)} samples")
                
                # Add label based on filename
                if 'fake' in name:
                    df['label'] = 'FALSE'
                    df['label_numeric'] = 0
                else:
                    df['label'] = 'TRUE'  
                    df['label_numeric'] = 1
                
                # Add source type
                if 'politifact' in name:
                    df['source_type'] = 'politifact'
                else:
                    df['source_type'] = 'gossipcop'
                
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded successfully")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"📊 Combined dataset: {len(combined_df)} total samples")
        logger.info(f"   TRUE samples: {(combined_df['label'] == 'TRUE').sum()}")
        logger.info(f"   FALSE samples: {(combined_df['label'] == 'FALSE').sum()}")
        
        # Enhanced preprocessing
        processed_df = self.preprocess_data(combined_df)
        
        # Create train/test split
        train_df, test_df = train_test_split(
            processed_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=processed_df['label_numeric']
        )
        
        logger.info(f"📈 Train set: {len(train_df)} samples")
        logger.info(f"📉 Test set: {len(test_df)} samples")
        
        return train_df, test_df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data preprocessing"""
        logger.info("🔧 Preprocessing data...")
        
        df = df.copy()
        
        # Clean and combine text fields
        df['text'] = df['title'].fillna('') + ' ' + df.get('text', '').fillna('')
        
        # Enhanced text cleaning
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Feature engineering
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        df['sentence_count'] = df['cleaned_text'].apply(lambda x: len(sent_tokenize(x)) if x else 0)
        df['avg_word_length'] = df['cleaned_text'].apply(self.avg_word_length)
        df['exclamation_count'] = df['text'].str.count('!')
        df['question_count'] = df['text'].str.count('\\?')
        df['caps_ratio'] = df['text'].apply(self.caps_ratio)
        
        # Remove samples with empty text
        df = df[df['cleaned_text'].str.len() > 10].reset_index(drop=True)
        
        logger.info(f"✅ Preprocessing complete: {len(df)} valid samples")
        return df

    def clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\\S+@\\S+', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^a-zA-Z0-9\\s]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        
        # Tokenize and lemmatize
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            return text.strip()

    def avg_word_length(self, text: str) -> float:
        """Calculate average word length"""
        if not text:
            return 0.0
        words = text.split()
        return np.mean([len(word) for word in words]) if words else 0.0

    def caps_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters"""
        if not text:
            return 0.0
        caps = sum(1 for c in text if c.isupper())
        return caps / len(text) if text else 0.0

    def train_advanced_models(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Train advanced ML models with FakeNewsNet data"""
        logger.info("🚀 Training advanced models on FakeNewsNet data...")
        
        X_train = train_df['cleaned_text']
        y_train = train_df['label_numeric']
        X_test = test_df['cleaned_text']
        y_test = test_df['label_numeric']
        
        results = {}
        
        # 1. Advanced TF-IDF + Ensemble
        logger.info("📊 Training TF-IDF + Ensemble models...")
        tfidf_results = self.train_tfidf_ensemble(X_train, y_train, X_test, y_test)
        results.update(tfidf_results)
        
        # 2. Character N-grams + Models
        logger.info("🔤 Training Character N-gram models...")
        char_results = self.train_char_ngram_models(X_train, y_train, X_test, y_test)
        results.update(char_results)
        
        # 3. Transformer-based models
        logger.info("🤖 Training Transformer models...")
        transformer_results = self.train_transformer_models(X_train, y_train, X_test, y_test)
        results.update(transformer_results)
        
        # 4. Ultimate Ensemble
        logger.info("🔥 Training Ultimate Ensemble...")
        ensemble_results = self.train_ultimate_ensemble(X_train, y_train, X_test, y_test, results)
        results.update(ensemble_results)
        
        # Save training report
        self.save_training_report(results, train_df, test_df)
        
        return results

    def train_tfidf_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train TF-IDF based ensemble models"""
        results = {}
        
        # Advanced TF-IDF Vectorizer
        tfidf = TfidfVectorizer(**self.model_configs['advanced_tfidf'])
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Save vectorizer
        joblib.dump(tfidf, self.models_dir / 'fakenews_tfidf_vectorizer.pkl')
        
        # Multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42),
            'logistic': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            'svm': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=150, max_depth=8, random_state=42, eval_metric='logloss')
        }
        
        for name, model in models.items():
            logger.info(f"   Training {name}...")
            
            # Train model
            model.fit(X_train_tfidf, y_train)
            
            # Evaluate
            train_score = model.score(X_train_tfidf, y_train)
            test_score = model.score(X_test_tfidf, y_test)
            y_pred = model.predict(X_test_tfidf)
            f1 = f1_score(y_test, y_pred)
            
            # Save model
            model_path = self.models_dir / f'fakenews_{name}_tfidf.pkl'
            joblib.dump(model, model_path)
            
            results[f'{name}_tfidf'] = {
                'model_path': str(model_path),
                'vectorizer_path': str(self.models_dir / 'fakenews_tfidf_vectorizer.pkl'),
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1,
                'model_type': 'tfidf'
            }
            
            logger.info(f"   {name}: Train={train_score:.3f}, Test={test_score:.3f}, F1={f1:.3f}")
        
        return results

    def train_char_ngram_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train character n-gram based models"""
        results = {}
        
        # Character N-gram Vectorizer
        char_vec = TfidfVectorizer(**self.model_configs['char_ngrams'])
        X_train_char = char_vec.fit_transform(X_train)
        X_test_char = char_vec.transform(X_test)
        
        # Save vectorizer
        joblib.dump(char_vec, self.models_dir / 'fakenews_char_vectorizer.pkl')
        
        # Best performing models for character n-grams
        models = {
            'random_forest': RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
            'logistic': LogisticRegression(max_iter=1000, C=0.5, random_state=42),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        for name, model in models.items():
            logger.info(f"   Training {name} (char n-grams)...")
            
            model.fit(X_train_char, y_train)
            
            train_score = model.score(X_train_char, y_train)
            test_score = model.score(X_test_char, y_test)
            y_pred = model.predict(X_test_char)
            f1 = f1_score(y_test, y_pred)
            
            model_path = self.models_dir / f'fakenews_{name}_char.pkl'
            joblib.dump(model, model_path)
            
            results[f'{name}_char'] = {
                'model_path': str(model_path),
                'vectorizer_path': str(self.models_dir / 'fakenews_char_vectorizer.pkl'),
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1,
                'model_type': 'char_ngram'
            }
            
            logger.info(f"   {name} (char): Train={train_score:.3f}, Test={test_score:.3f}, F1={f1:.3f}")
        
        return results

    def train_transformer_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train transformer-based models"""
        results = {}
        
        try:
            logger.info("   Training Sentence-BERT embeddings...")
            
            # Use sentence-transformers for embeddings
            model_name = 'all-MiniLM-L6-v2'  # Lightweight but effective
            sentence_model = SentenceTransformer(model_name)
            
            # Generate embeddings (in batches to avoid memory issues)
            batch_size = 32
            
            def get_embeddings_batch(texts):
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts.iloc[i:i+batch_size].tolist()
                    batch_embeddings = sentence_model.encode(batch, convert_to_numpy=True)
                    embeddings.append(batch_embeddings)
                return np.vstack(embeddings)
            
            logger.info("   Generating training embeddings...")
            X_train_emb = get_embeddings_batch(X_train)
            
            logger.info("   Generating test embeddings...")  
            X_test_emb = get_embeddings_batch(X_test)
            
            # Train classifiers on embeddings
            models = {
                'logistic': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
            }
            
            for name, model in models.items():
                logger.info(f"   Training {name} (SBERT embeddings)...")
                
                model.fit(X_train_emb, y_train)
                
                train_score = model.score(X_train_emb, y_train)
                test_score = model.score(X_test_emb, y_test)
                y_pred = model.predict(X_test_emb)
                f1 = f1_score(y_test, y_pred)
                
                model_path = self.models_dir / f'fakenews_{name}_sbert.pkl'
                joblib.dump(model, model_path)
                
                results[f'{name}_sbert'] = {
                    'model_path': str(model_path),
                    'vectorizer_path': model_name,  # Store the sentence-transformer model name
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'f1_score': f1,
                    'model_type': 'transformer'
                }
                
                logger.info(f"   {name} (SBERT): Train={train_score:.3f}, Test={test_score:.3f}, F1={f1:.3f}")
        
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            logger.info("Continuing without transformer models...")
        
        return results

    def train_ultimate_ensemble(self, X_train, y_train, X_test, y_test, all_results: Dict) -> Dict:
        """Train ultimate ensemble combining best models"""
        logger.info("   Creating Ultimate Ensemble...")
        
        # Select top models based on F1 score
        top_models = sorted(
            [(name, info) for name, info in all_results.items() if 'f1_score' in info],
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )[:5]  # Top 5 models
        
        logger.info(f"   Selected top {len(top_models)} models for ensemble:")
        for name, info in top_models:
            logger.info(f"     - {name}: F1={info['f1_score']:.3f}")
        
        # Load and prepare models for ensemble
        ensemble_models = []
        model_names = []
        
        for name, info in top_models:
            try:
                model = joblib.load(info['model_path'])
                
                # Get predictions based on model type
                if info['model_type'] == 'tfidf':
                    vectorizer = joblib.load(info['vectorizer_path'])
                    X_train_vec = vectorizer.transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                elif info['model_type'] == 'char_ngram':
                    vectorizer = joblib.load(info['vectorizer_path'])
                    X_train_vec = vectorizer.transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                elif info['model_type'] == 'transformer':
                    # For transformer models, we'd need to regenerate embeddings
                    # Skip for now to avoid recomputation
                    continue
                
                # Create a pipeline for this model
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
                
                ensemble_models.append((name, pipeline))
                model_names.append(name)
                
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                continue
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble")
            return {}
        
        # Create voting ensemble
        voting_classifier = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'
        )
        
        # Train ensemble
        voting_classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = voting_classifier.score(X_train, y_train)
        test_score = voting_classifier.score(X_test, y_test)
        y_pred = voting_classifier.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        # Save ensemble
        ensemble_path = self.models_dir / 'fakenews_ultimate_ensemble.pkl'
        joblib.dump(voting_classifier, ensemble_path)
        
        logger.info(f"   Ultimate Ensemble: Train={train_score:.3f}, Test={test_score:.3f}, F1={f1:.3f}")
        
        return {
            'ultimate_ensemble': {
                'model_path': str(ensemble_path),
                'component_models': model_names,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'f1_score': f1,
                'model_type': 'ensemble'
            }
        }

    def save_training_report(self, results: Dict, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save comprehensive training report"""
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(train_df) + len(test_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'true_samples': int((train_df['label_numeric'] == 1).sum() + (test_df['label_numeric'] == 1).sum()),
                'false_samples': int((train_df['label_numeric'] == 0).sum() + (test_df['label_numeric'] == 0).sum())
            },
            'model_results': results,
            'best_models': {
                'best_f1': max(results.items(), key=lambda x: x[1].get('f1_score', 0))[0] if results else None,
                'best_accuracy': max(results.items(), key=lambda x: x[1].get('test_accuracy', 0))[0] if results else None
            },
            'training_config': {
                'tfidf_config': self.model_configs['advanced_tfidf'],
                'char_config': self.model_configs['char_ngrams'],
                'data_source': 'FakeNewsNet (PolitiFact + GossipCop)'
            }
        }
        
        # Save report
        report_path = self.models_dir / 'fakenews_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Training report saved: {report_path}")
        
        # Print summary
        self.print_training_summary(results)

    def print_training_summary(self, results: Dict):
        """Print training summary"""
        print("\\n" + "="*80)
        print("🎯 FAKENEWSNET TRAINING SUMMARY")
        print("="*80)
        
        if not results:
            print("❌ No models trained successfully")
            return
        
        # Sort by F1 score
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].get('f1_score', 0), 
            reverse=True
        )
        
        print("📊 MODEL PERFORMANCE RANKING:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<25} {'Test Accuracy':<12} {'F1 Score':<10} {'Type':<15}")
        print("-" * 80)
        
        for i, (name, metrics) in enumerate(sorted_results, 1):
            print(f"{i:<4} {name:<25} {metrics.get('test_accuracy', 0):<12.3f} {metrics.get('f1_score', 0):<10.3f} {metrics.get('model_type', 'unknown'):<15}")
        
        # Best model info
        best_model = sorted_results[0]
        print("\\n🏆 BEST PERFORMING MODEL:")
        print(f"   Model: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.3f}")
        print(f"   F1 Score: {best_model[1]['f1_score']:.3f}")
        print(f"   Model Path: {best_model[1]['model_path']}")
        
        print("\\n✅ All models saved to trained_models/ directory")
        print("🔄 Ready for integration with TruthMate service!")
        print("="*80)

    def integrate_with_truthmate(self):
        """Integrate best models with existing TruthMate service"""
        logger.info("🔗 Integrating FakeNewsNet models with TruthMate...")
        
        # Load training report to find best models
        report_path = self.models_dir / 'fakenews_training_report.json'
        if not report_path.exists():
            logger.error("Training report not found. Please run training first.")
            return False
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        best_models = report.get('best_models', {})
        if not best_models:
            logger.error("No best models found in report")
            return False
        
        # Create integration script
        integration_script = self.create_integration_script(report)
        
        script_path = self.models_dir / 'integrate_fakenews_models.py'
        with open(script_path, 'w') as f:
            f.write(integration_script)
        
        logger.info(f"📜 Integration script created: {script_path}")
        logger.info("🚀 Run the integration script to update TruthMate service")
        
        return True

    def create_integration_script(self, report: Dict) -> str:
        """Create integration script for TruthMate service"""
        best_f1_model = report['best_models']['best_f1']
        
        script = f'''"""
FakeNewsNet Model Integration Script for TruthMate
Auto-generated integration code
"""
import joblib
import numpy as np
from pathlib import Path

class FakeNewsNetIntegration:
    """Integration class for FakeNewsNet trained models"""
    
    def __init__(self, models_dir="trained_models"):
        self.models_dir = Path(models_dir)
        self.best_model = None
        self.best_vectorizer = None
        self.load_best_model()
    
    def load_best_model(self):
        """Load the best performing FakeNewsNet model"""
        try:
            # Best model: {best_f1_model}
            model_path = self.models_dir / "fakenews_{best_f1_model.split('_')[0]}_{{model_type}}.pkl"
            vectorizer_path = self.models_dir / "fakenews_{{model_type}}_vectorizer.pkl"
            
            # Try different model types
            model_types = ['tfidf', 'char', 'sbert']
            
            for model_type in model_types:
                try:
                    model_file = str(model_path).format(model_type=model_type)
                    vec_file = str(vectorizer_path).format(model_type=model_type)
                    
                    if Path(model_file).exists() and Path(vec_file).exists():
                        self.best_model = joblib.load(model_file)
                        self.best_vectorizer = joblib.load(vec_file)
                        print(f"✅ Loaded FakeNewsNet model: {{model_type}}")
                        return True
                except Exception as e:
                    continue
            
            print("⚠️ Could not load FakeNewsNet models, using fallback")
            return False
            
        except Exception as e:
            print(f"❌ Error loading FakeNewsNet models: {{e}}")
            return False
    
    def predict(self, text: str) -> dict:
        """Predict using FakeNewsNet model"""
        if not self.best_model or not self.best_vectorizer:
            return {{'verdict': 'UNKNOWN', 'confidence': 0.5, 'method': 'FALLBACK'}}
        
        try:
            # Vectorize text
            text_vec = self.best_vectorizer.transform([text])
            
            # Get prediction
            prediction = self.best_model.predict(text_vec)[0]
            
            # Get confidence if available
            if hasattr(self.best_model, 'predict_proba'):
                confidence = float(np.max(self.best_model.predict_proba(text_vec)[0]))
            else:
                confidence = 0.75
            
            # Convert to TruthMate format
            verdict = 'TRUE' if prediction == 1 else 'FALSE'
            
            return {{
                'verdict': verdict,
                'confidence': confidence,
                'method': 'FAKENEWSNET_ML'
            }}
            
        except Exception as e:
            print(f"FakeNewsNet prediction error: {{e}}")
            return {{'verdict': 'UNKNOWN', 'confidence': 0.5, 'method': 'ERROR'}}

# Integration instructions:
# 1. Copy this class to your ultimate_working_service.py
# 2. Initialize in UltimateWorkingFactChecker.__init__():
#    self.fakenews_model = FakeNewsNetIntegration()
# 3. Use in _analyze_with_ml() method:
#    fakenews_result = self.fakenews_model.predict(claim)
#    # Combine with existing ML analysis

print("🔗 FakeNewsNet Integration Ready!")
print("📝 Follow the integration instructions above")
'''
        
        return script

    def run_full_pipeline(self):
        """Run the complete FakeNewsNet training pipeline"""
        logger.info("🚀 Starting FakeNewsNet Training Pipeline for TruthMate")
        
        try:
            # Step 1: Download data
            if not self.download_fakenews_data():
                logger.error("❌ Data download failed")
                return False
            
            # Step 2: Load and prepare data  
            train_df, test_df = self.load_and_prepare_data()
            
            # Step 3: Train models
            results = self.train_advanced_models(train_df, test_df)
            
            if not results:
                logger.error("❌ No models trained successfully")
                return False
            
            # Step 4: Create integration
            self.integrate_with_truthmate()
            
            logger.info("🎉 FakeNewsNet training pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False

def main():
    """Main training function"""
    print("="*80)
    print("🤖 FakeNewsNet Training Pipeline for TruthMate")
    print("="*80)
    
    trainer = FakeNewsNetTrainer()
    success = trainer.run_full_pipeline()
    
    if success:
        print("\\n✅ Training completed successfully!")
        print("🔄 Check the integration script in trained_models/")
        print("📊 View training report: trained_models/fakenews_training_report.json")
    else:
        print("\\n❌ Training failed!")
        print("📋 Check fakenews_training.log for details")

if __name__ == "__main__":
    main()