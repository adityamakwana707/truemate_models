"""
TruthMate Model Training - Production Version
Train and fine-tune fact-checking models for maximum accuracy
"""
import os
import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthMateTrainer:
    def __init__(self):
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
        except ImportError:
            self.device = 'cpu'
            logger.info("PyTorch not available, using CPU-only mode")
        
        self.training_data = []
        self.test_data = []
        
    def prepare_training_data(self):
        """Prepare comprehensive training data"""
        logger.info("🔄 Preparing training data...")
        
        # High-quality curated dataset
        training_samples = [
            # COVID-19 and Health Facts
            {
                'claim': 'COVID-19 vaccines are safe and effective according to clinical trials',
                'label': 'True',
                'confidence': 0.95,
                'category': 'health',
                'explanation': 'Multiple Phase 3 clinical trials with over 40,000 participants each demonstrated high efficacy and acceptable safety profiles.'
            },
            {
                'claim': 'COVID-19 vaccines contain microchips for tracking people',
                'label': 'False', 
                'confidence': 0.98,
                'category': 'health',
                'explanation': 'This is a debunked conspiracy theory. Vaccines contain mRNA or viral proteins, not electronic devices.'
            },
            {
                'claim': 'Drinking bleach can cure COVID-19',
                'label': 'False',
                'confidence': 0.99,
                'category': 'health',
                'explanation': 'Bleach is a toxic substance that can cause severe internal damage and death. It has no medicinal properties.'
            },
            {
                'claim': 'Natural immunity from COVID-19 is always better than vaccine immunity',
                'label': 'Misleading',
                'confidence': 0.85,
                'category': 'health', 
                'explanation': 'Natural immunity varies greatly between individuals and wanes over time. Vaccination provides more consistent protection.'
            },
            
            # Climate Science
            {
                'claim': 'Human activities are the primary driver of current climate change',
                'label': 'True',
                'confidence': 0.97,
                'category': 'climate',
                'explanation': 'Overwhelming scientific consensus based on multiple lines of evidence confirms anthropogenic climate change.'
            },
            {
                'claim': 'Climate change is a hoax created by scientists for funding',
                'label': 'False',
                'confidence': 0.96,
                'category': 'climate',
                'explanation': 'Climate change is supported by evidence from thousands of independent researchers worldwide across multiple institutions.'
            },
            {
                'claim': 'CO2 is plant food so more CO2 is always beneficial',
                'label': 'Misleading',
                'confidence': 0.88,
                'category': 'climate',
                'explanation': 'While plants use CO2, higher concentrations also cause warming and extreme weather that can harm plant growth.'
            },
            
            # Technology
            {
                'claim': '5G networks cause COVID-19 and cancer',
                'label': 'False',
                'confidence': 0.98,
                'category': 'technology',
                'explanation': '5G uses non-ionizing radio waves that cannot damage DNA or cause viral infections. No credible scientific evidence supports these claims.'
            },
            {
                'claim': 'Artificial intelligence will replace all human jobs within 10 years',
                'label': 'Misleading',
                'confidence': 0.82,
                'category': 'technology',
                'explanation': 'While AI will automate some jobs, experts predict gradual change with new job creation. Complete replacement is unlikely in 10 years.'
            },
            
            # Politics and Social Issues
            {
                'claim': 'Voter fraud determined the outcome of the 2020 US presidential election',
                'label': 'False',
                'confidence': 0.92,
                'category': 'politics',
                'explanation': 'Multiple audits, recounts, and court cases found no evidence of fraud sufficient to change the election outcome.'
            },
            {
                'claim': 'Social media platforms have some bias in content moderation',
                'label': 'True',
                'confidence': 0.78,
                'category': 'politics',
                'explanation': 'Studies and reports indicate various forms of bias in content moderation, though the extent and nature are debated.'
            },
            
            # Economics
            {
                'claim': 'Minimum wage increases always reduce employment',
                'label': 'Misleading',
                'confidence': 0.75,
                'category': 'economics',
                'explanation': 'Economic research shows mixed results. Some studies find small employment effects, others find minimal impact.'
            },
            {
                'claim': 'Cryptocurrency has no environmental impact',
                'label': 'False',
                'confidence': 0.89,
                'category': 'economics',
                'explanation': 'Bitcoin and other proof-of-work cryptocurrencies consume significant electricity for mining operations.'
            },
            
            # General Science
            {
                'claim': 'The Earth is flat and space agencies are lying',
                'label': 'False',
                'confidence': 0.99,
                'category': 'science',
                'explanation': 'The spherical Earth is supported by overwhelming evidence from physics, astronomy, geography, and direct observation.'
            },
            {
                'claim': 'Regular exercise improves cardiovascular health',
                'label': 'True',
                'confidence': 0.98,
                'category': 'health',
                'explanation': 'Extensive medical research consistently shows exercise reduces heart disease risk and improves overall health.'
            },
            
            # Nutrition and Health
            {
                'claim': 'Moderate coffee consumption may reduce heart disease risk',
                'label': 'True',
                'confidence': 0.83,
                'category': 'health',
                'explanation': 'Multiple studies suggest 3-4 cups of coffee daily may be associated with reduced cardiovascular disease risk.'
            },
            {
                'claim': 'All vaccines cause autism',
                'label': 'False',
                'confidence': 0.96,
                'category': 'health',
                'explanation': 'Extensive research has found no causal link between vaccines and autism. The original study was fraudulent and retracted.'
            },
        ]
        
        # Convert labels to numerical format for training
        label_mapping = {'True': 2, 'False': 0, 'Misleading': 1}
        
        for sample in training_samples:
            sample['label_num'] = label_mapping[sample['label']]
        
        self.training_data = training_samples
        logger.info(f"✅ Prepared {len(training_samples)} training samples")
        
        # Create test data (20% split)
        test_size = int(0.2 * len(training_samples))
        self.test_data = training_samples[:test_size]
        self.training_data = training_samples[test_size:]
        
        logger.info(f"📊 Training: {len(self.training_data)}, Testing: {len(self.test_data)}")
    
    def train_lightweight_models(self):
        """Train lightweight models that work without heavy dependencies"""
        logger.info("🧠 Training lightweight fact-checking models...")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.model_selection import cross_val_score
            import joblib
            
            # Prepare data
            claims = [item['claim'] for item in self.training_data]
            labels = [item['label_num'] for item in self.training_data]
            
            test_claims = [item['claim'] for item in self.test_data]
            test_labels = [item['label_num'] for item in self.test_data]
            
            # Create TF-IDF features
            logger.info("📝 Creating TF-IDF features...")
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True
            )
            
            X_train = vectorizer.fit_transform(claims)
            X_test = vectorizer.transform(test_claims)
            
            # Train Random Forest
            logger.info("🌳 Training Random Forest classifier...")
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            rf_model.fit(X_train, labels)
            rf_pred = rf_model.predict(X_test)
            rf_accuracy = accuracy_score(test_labels, rf_pred)
            
            logger.info(f"🎯 Random Forest Accuracy: {rf_accuracy:.3f}")
            
            # Train Logistic Regression
            logger.info("📈 Training Logistic Regression...")
            lr_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            
            lr_model.fit(X_train, labels)
            lr_pred = lr_model.predict(X_test)
            lr_accuracy = accuracy_score(test_labels, lr_pred)
            
            logger.info(f"🎯 Logistic Regression Accuracy: {lr_accuracy:.3f}")
            
            # Cross-validation
            rf_cv_scores = cross_val_score(rf_model, X_train, labels, cv=5)
            lr_cv_scores = cross_val_score(lr_model, X_train, labels, cv=5)
            
            logger.info(f"🔄 Random Forest CV: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
            logger.info(f"🔄 Logistic Regression CV: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std() * 2:.3f})")
            
            # Save models
            os.makedirs('trained_models', exist_ok=True)
            
            joblib.dump(vectorizer, 'trained_models/tfidf_vectorizer.pkl')
            joblib.dump(rf_model, 'trained_models/random_forest_model.pkl')
            joblib.dump(lr_model, 'trained_models/logistic_regression_model.pkl')
            
            logger.info("💾 Models saved to trained_models/ directory")
            
            # Generate detailed report
            self._generate_training_report(rf_model, lr_model, vectorizer, test_claims, test_labels, rf_pred, lr_pred)
            
            return {
                'rf_accuracy': rf_accuracy,
                'lr_accuracy': lr_accuracy,
                'rf_cv_mean': rf_cv_scores.mean(),
                'lr_cv_mean': lr_cv_scores.mean()
            }
            
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            logger.info("Install with: pip install scikit-learn joblib")
            return None
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def _generate_training_report(self, rf_model, lr_model, vectorizer, test_texts, test_labels, rf_pred, lr_pred):
        """Generate comprehensive training report"""
        
        try:
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            
            # Create report
            report = {
                'timestamp': datetime.now().isoformat(),
                'training_summary': {
                    'total_samples': len(self.training_data),
                    'test_samples': len(self.test_data),
                    'categories': list(set(item['category'] for item in self.training_data))
                },
                'model_performance': {
                    'random_forest': {
                        'accuracy': accuracy_score(test_labels, rf_pred),
                        'classification_report': classification_report(test_labels, rf_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(test_labels, rf_pred).tolist()
                    },
                    'logistic_regression': {
                        'accuracy': accuracy_score(test_labels, lr_pred), 
                        'classification_report': classification_report(test_labels, lr_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(test_labels, lr_pred).tolist()
                    }
                }
            }
            
            # Save report
            with open('trained_models/training_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Test predictions on sample claims
            sample_claims = [
                "COVID-19 vaccines are dangerous and should be avoided",
                "Exercise improves heart health",
                "The Earth is flat",
                "Climate change is mostly natural"
            ]
            
            print("\n🧪 Sample Predictions:")
            print("=" * 50)
            
            X_sample = vectorizer.transform(sample_claims)
            rf_sample_pred = rf_model.predict(X_sample)
            lr_sample_pred = lr_model.predict(X_sample)
            
            label_names = ['False', 'Misleading', 'True']
            
            for i, claim in enumerate(sample_claims):
                print(f"\nClaim: {claim}")
                print(f"RF Prediction: {label_names[rf_sample_pred[i]]}")
                print(f"LR Prediction: {label_names[lr_sample_pred[i]]}")
            
            logger.info("📊 Training report saved to trained_models/training_report.json")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

def main():
    """Main training execution"""
    print("🎓 TruthMate Model Training Pipeline")
    print("=" * 50)
    
    trainer = TruthMateTrainer()
    
    # Prepare training data
    trainer.prepare_training_data()
    
    # Train lightweight models
    print("\n🧠 Training lightweight models...")
    lightweight_results = trainer.train_lightweight_models()
    
    if lightweight_results:
        print(f"\n📊 Training Results:")
        print(f"Random Forest Accuracy: {lightweight_results['rf_accuracy']:.3f}")
        print(f"Logistic Regression Accuracy: {lightweight_results['lr_accuracy']:.3f}")
        print(f"Random Forest CV Score: {lightweight_results['rf_cv_mean']:.3f}")
        print(f"Logistic Regression CV Score: {lightweight_results['lr_cv_mean']:.3f}")
        
        print("\n🎉 Training complete!")
        print("📍 Models saved to trained_models/ directory")
        print("🚀 Use the production service to test your trained models")
    else:
        print("\n❌ Training failed - check dependencies")
        print("Install required packages: pip install scikit-learn joblib")

if __name__ == "__main__":
    main()