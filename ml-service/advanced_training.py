"""
Advanced Model Training Pipeline for TruthMate
State-of-the-art training with multiple datasets and techniques
"""
import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, AdamW
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import wandb
from datasets import Dataset as HFDataset
import argparse
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFactCheckDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Data augmentation techniques
        if self.augment and np.random.random() > 0.7:
            text = self.augment_text(text)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def augment_text(self, text):
        """Simple data augmentation techniques"""
        # Random word order shuffling for robustness
        words = text.split()
        if len(words) > 3 and np.random.random() > 0.8:
            # Shuffle middle words, keep first and last
            middle = words[1:-1]
            np.random.shuffle(middle)
            return ' '.join([words[0]] + middle + [words[-1]])
        return text

class AdvancedModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_multiple_datasets(self, dataset_paths: Dict[str, str]) -> Tuple[List[str], List[int]]:
        """Load and combine multiple datasets"""
        all_texts = []
        all_labels = []
        dataset_sources = []
        
        label_mapping = {'True': 2, 'False': 0, 'Misleading': 1, 'Unknown': 1}
        
        for dataset_name, path in dataset_paths.items():
            logger.info(f"Loading {dataset_name} from {path}")
            
            try:
                if dataset_name == 'liar':
                    texts, labels, sources = self.load_liar_dataset(path)
                elif dataset_name == 'fever':
                    texts, labels, sources = self.load_fever_dataset(path)
                elif dataset_name == 'fakenews':
                    texts, labels, sources = self.load_fakenews_dataset(path)
                elif dataset_name == 'snopes':
                    texts, labels, sources = self.load_snopes_dataset(path)
                elif dataset_name == 'politifact':
                    texts, labels, sources = self.load_politifact_dataset(path)
                else:
                    logger.warning(f"Unknown dataset: {dataset_name}")
                    continue
                
                # Convert to standardized labels
                standardized_labels = []
                for label in labels:
                    if isinstance(label, str):
                        standardized_labels.append(label_mapping.get(label, 1))
                    else:
                        standardized_labels.append(int(label))
                
                all_texts.extend(texts)
                all_labels.extend(standardized_labels)
                dataset_sources.extend([dataset_name] * len(texts))
                
                logger.info(f"Loaded {len(texts)} samples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        
        logger.info(f"Total samples loaded: {len(all_texts)}")
        logger.info(f"Label distribution: {Counter(all_labels)}")
        
        return all_texts, all_labels, dataset_sources
    
    def load_liar_dataset(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load LIAR dataset with enhanced processing"""
        try:
            # Load all splits
            splits = ['train.tsv', 'valid.tsv', 'test.tsv']
            all_data = []
            
            for split in splits:
                file_path = os.path.join(path, split)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, sep='\t', header=None)
                    all_data.append(df)
            
            if not all_data:
                raise FileNotFoundError("No LIAR dataset files found")
            
            df = pd.concat(all_data, ignore_index=True)
            
            # LIAR format: [label, statement, subject, speaker, job, state, party, ...]
            texts = df[1].astype(str).tolist()  # Statement
            raw_labels = df[0].astype(str).tolist()  # Labels
            
            # Convert LIAR labels to our format
            label_conversion = {
                'pants-fire': 'False',
                'false': 'False', 
                'barely-true': 'False',
                'half-true': 'Misleading',
                'mostly-true': 'True',
                'true': 'True'
            }
            
            labels = [label_conversion.get(label.lower(), 'Unknown') for label in raw_labels]
            sources = ['liar'] * len(texts)
            
            # Enhanced text with context
            if len(df.columns) > 2:
                subjects = df[2].fillna('').astype(str)
                speakers = df[3].fillna('').astype(str)
                
                enhanced_texts = []
                for i, (text, subject, speaker) in enumerate(zip(texts, subjects, speakers)):
                    enhanced_text = text
                    if subject and subject != 'nan':
                        enhanced_text += f" [Subject: {subject}]"
                    if speaker and speaker != 'nan':
                        enhanced_text += f" [Speaker: {speaker}]"
                    enhanced_texts.append(enhanced_text)
                
                texts = enhanced_texts
            
            return texts, labels, sources
            
        except Exception as e:
            logger.error(f"Error loading LIAR dataset: {e}")
            return [], [], []
    
    def load_fever_dataset(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load FEVER dataset"""
        try:
            texts = []
            labels = []
            
            # Load FEVER JSONL files
            for filename in ['train.jsonl', 'dev.jsonl']:
                file_path = os.path.join(path, filename)
                if not os.path.exists(file_path):
                    continue
                    
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            claim = data.get('claim', '')
                            label = data.get('label', 'NOT ENOUGH INFO')
                            
                            if claim and label in ['SUPPORTS', 'REFUTES']:
                                texts.append(claim)
                                labels.append('True' if label == 'SUPPORTS' else 'False')
                        except:
                            continue
            
            sources = ['fever'] * len(texts)
            return texts, labels, sources
            
        except Exception as e:
            logger.error(f"Error loading FEVER dataset: {e}")
            return [], [], []
    
    def load_fakenews_dataset(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load FakeNews dataset"""
        try:
            df = pd.read_csv(path)
            
            # Handle different possible column names
            text_cols = ['text', 'content', 'article', 'news']
            label_cols = ['label', 'class', 'fake', 'real']
            
            text_col = None
            label_col = None
            
            for col in text_cols:
                if col in df.columns:
                    text_col = col
                    break
                    
            for col in label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if not text_col or not label_col:
                raise ValueError("Could not identify text and label columns")
            
            texts = df[text_col].astype(str).tolist()
            raw_labels = df[label_col].tolist()
            
            # Convert labels to our format
            labels = []
            for label in raw_labels:
                if isinstance(label, str):
                    if label.lower() in ['fake', 'false', '0']:
                        labels.append('False')
                    elif label.lower() in ['real', 'true', '1']:
                        labels.append('True')
                    else:
                        labels.append('Unknown')
                else:
                    labels.append('True' if int(label) == 1 else 'False')
            
            sources = ['fakenews'] * len(texts)
            return texts, labels, sources
            
        except Exception as e:
            logger.error(f"Error loading FakeNews dataset: {e}")
            return [], [], []
    
    def load_snopes_dataset(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load Snopes fact-check dataset"""
        # Implementation for Snopes dataset
        # This would be similar to other loaders
        return [], [], []
    
    def load_politifact_dataset(self, path: str) -> Tuple[List[str], List[str], List[str]]:
        """Load PolitiFact dataset"""
        # Implementation for PolitiFact dataset
        return [], [], []
    
    def create_balanced_dataset(self, texts: List[str], labels: List[int], 
                              sources: List[str]) -> Tuple[List[str], List[int], List[str]]:
        """Create balanced dataset using oversampling/undersampling"""
        
        # Count samples per class
        label_counts = Counter(labels)
        logger.info(f"Original distribution: {label_counts}")
        
        # Find target size (median of class counts)
        target_size = int(np.median(list(label_counts.values())))
        target_size = max(target_size, 1000)  # Minimum 1000 per class
        
        balanced_texts = []
        balanced_labels = []
        balanced_sources = []
        
        for label in set(labels):
            # Get all samples for this label
            label_indices = [i for i, l in enumerate(labels) if l == label]
            label_texts = [texts[i] for i in label_indices]
            label_sources = [sources[i] for i in label_indices]
            
            current_size = len(label_texts)
            
            if current_size >= target_size:
                # Undersample
                selected_indices = np.random.choice(current_size, target_size, replace=False)
                selected_texts = [label_texts[i] for i in selected_indices]
                selected_sources = [label_sources[i] for i in selected_indices]
            else:
                # Oversample
                oversample_indices = np.random.choice(current_size, target_size, replace=True)
                selected_texts = [label_texts[i] for i in oversample_indices]
                selected_sources = [label_sources[i] for i in oversample_indices]
            
            balanced_texts.extend(selected_texts)
            balanced_labels.extend([label] * target_size)
            balanced_sources.extend(selected_sources)
        
        # Shuffle the balanced dataset
        combined = list(zip(balanced_texts, balanced_labels, balanced_sources))
        np.random.shuffle(combined)
        balanced_texts, balanced_labels, balanced_sources = zip(*combined)
        
        logger.info(f"Balanced distribution: {Counter(balanced_labels)}")
        
        return list(balanced_texts), list(balanced_labels), list(balanced_sources)
    
    def train_advanced_model(self, texts: List[str], labels: List[int], 
                           model_name: str = 'microsoft/deberta-v3-base',
                           output_dir: str = './models/advanced_fact_checker'):
        """Train advanced fact-checking model with all optimizations"""
        
        logger.info(f"Training advanced model: {model_name}")
        
        # Split dataset with stratification
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        logger.info(f"Dataset splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # True, False, Misleading
            problem_type="single_label_classification"
        )
        
        # Add special tokens if needed
        special_tokens = ['[CLAIM]', '[EVIDENCE]', '[SOURCE]']
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        # Create datasets with augmentation
        train_dataset = AdvancedFactCheckDataset(train_texts, train_labels, tokenizer, augment=True)
        val_dataset = AdvancedFactCheckDataset(val_texts, val_labels, tokenizer, augment=False)
        test_dataset = AdvancedFactCheckDataset(test_texts, test_labels, tokenizer, augment=False)
        
        # Compute class weights for imbalanced dataset
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Advanced training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps", 
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",  # Disable wandb for now
            save_total_limit=3,
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
            group_by_length=True,
            lr_scheduler_type="cosine",
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
        )
        
        # Custom trainer with class weights
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_advanced_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting advanced training...")
        train_result = trainer.train()
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        logger.info(f"Test results: {test_results}")
        
        # Save model and tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save training metrics
        with open(f'{output_dir}/training_results.json', 'w') as f:
            json.dump({
                'train_results': train_result.metrics,
                'test_results': test_results,
                'model_name': model_name,
                'dataset_size': len(texts),
                'class_distribution': dict(Counter(labels))
            }, f, indent=2)
        
        # Generate detailed evaluation
        self.generate_detailed_evaluation(trainer, test_dataset, test_labels, output_dir)
        
        logger.info(f"Advanced model training completed! Saved to {output_dir}")
        
        return trainer, model, tokenizer
    
    def compute_advanced_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'f1_class_2': f1_per_class[2] if len(f1_per_class) > 2 else 0,
        }
    
    def generate_detailed_evaluation(self, trainer, test_dataset, test_labels, output_dir):
        """Generate comprehensive evaluation report"""
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        # Label names
        label_names = ['False', 'Misleading', 'True']
        
        # Classification report
        report = classification_report(
            test_labels, pred_labels,
            target_names=label_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, pred_labels)
        
        # Save detailed report
        with open(f'{output_dir}/detailed_evaluation.json', 'w') as f:
            json.dump({
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'label_names': label_names
            }, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix - Advanced Fact Checker')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Detailed evaluation saved!")

def main():
    parser = argparse.ArgumentParser(description='Train advanced fact-checking models')
    
    # Dataset arguments
    parser.add_argument('--liar_path', type=str, help='Path to LIAR dataset directory')
    parser.add_argument('--fever_path', type=str, help='Path to FEVER dataset directory')
    parser.add_argument('--fakenews_path', type=str, help='Path to FakeNews CSV file')
    parser.add_argument('--snopes_path', type=str, help='Path to Snopes dataset')
    parser.add_argument('--politifact_path', type=str, help='Path to PolitiFact dataset')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                       choices=[
                           'microsoft/deberta-v3-base',
                           'microsoft/deberta-v3-large', 
                           'roberta-large',
                           'bert-large-uncased',
                           'facebook/bart-large'
                       ],
                       help='Pre-trained model to fine-tune')
    
    parser.add_argument('--output_dir', type=str, default='./models/advanced_fact_checker',
                       help='Output directory for trained model')
    
    # Training arguments
    parser.add_argument('--balance_dataset', action='store_true',
                       help='Balance dataset using over/undersampling')
    parser.add_argument('--use_sample', action='store_true',
                       help='Use sample data if no datasets provided')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Prepare dataset paths
    dataset_paths = {}
    if args.liar_path:
        dataset_paths['liar'] = args.liar_path
    if args.fever_path:
        dataset_paths['fever'] = args.fever_path
    if args.fakenews_path:
        dataset_paths['fakenews'] = args.fakenews_path
    if args.snopes_path:
        dataset_paths['snopes'] = args.snopes_path
    if args.politifact_path:
        dataset_paths['politifact'] = args.politifact_path
    
    # Load datasets
    if dataset_paths:
        texts, labels, sources = trainer.load_multiple_datasets(dataset_paths)
    elif args.use_sample:
        logger.info("Using sample dataset for testing...")
        # Create sample data
        sample_data = [
            ("COVID-19 vaccines are safe and effective", 2),  # True
            ("The Earth is flat", 0),  # False
            ("Climate change might be natural", 1),  # Misleading
            ("Exercise improves health", 2),  # True
            ("5G causes coronavirus", 0),  # False
        ] * 200  # Repeat for sufficient data
        
        texts, labels = zip(*sample_data)
        texts, labels = list(texts), list(labels)
        sources = ['sample'] * len(texts)
    else:
        logger.error("No datasets provided! Use dataset paths or --use_sample flag")
        return
    
    if not texts:
        logger.error("No data loaded!")
        return
    
    # Balance dataset if requested
    if args.balance_dataset and len(set(labels)) > 1:
        texts, labels, sources = trainer.create_balanced_dataset(texts, labels, sources)
    
    # Train advanced model
    trainer_obj, model, tokenizer = trainer.train_advanced_model(
        texts=texts,
        labels=labels,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    logger.info("Advanced training pipeline completed!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("Ready for deployment!")

if __name__ == "__main__":
    main()

# Usage examples:
# python advanced_training.py --use_sample --model_name microsoft/deberta-v3-base
# python advanced_training.py --liar_path ./data/liar --fever_path ./data/fever --balance_dataset
# python advanced_training.py --fakenews_path ./data/fake_news.csv --model_name roberta-large