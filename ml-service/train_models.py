"""
Training Script for Enhanced Fact-Checking Models
This script helps you train custom models on fact-checking datasets
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import argparse
from datasets import Dataset as HFDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactCheckDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
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

def load_liar_dataset(data_path):
    """Load and preprocess LIAR dataset"""
    try:
        # LIAR dataset format: label, statement, subject, speaker, job, state, party, etc.
        train_df = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t', header=None)
        valid_df = pd.read_csv(os.path.join(data_path, 'valid.tsv'), sep='\t', header=None)
        test_df = pd.read_csv(os.path.join(data_path, 'test.tsv'), sep='\t', header=None)
        
        # Combine all datasets
        df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
        
        # Extract text (column 1) and labels (column 0)
        texts = df[1].tolist()  # Statement text
        labels_text = df[0].tolist()  # Truth labels
        
        # Convert labels to binary or multi-class
        # Option 1: Binary (True/False)
        label_mapping_binary = {
            'pants-fire': 0, 'false': 0, 'barely-true': 0,  # False
            'half-true': 1, 'mostly-true': 1, 'true': 1     # True
        }
        
        # Option 2: Multi-class (6 classes)
        label_mapping_multi = {
            'pants-fire': 0, 'false': 1, 'barely-true': 2,
            'half-true': 3, 'mostly-true': 4, 'true': 5
        }
        
        # Use binary classification for simplicity
        labels = [label_mapping_binary.get(label, 0) for label in labels_text]
        
        # Filter out any None labels
        valid_indices = [i for i, label in enumerate(labels) if label is not None]
        texts = [texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        logger.info(f"Loaded LIAR dataset: {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Error loading LIAR dataset: {e}")
        return [], []

def load_fever_dataset(data_path):
    """Load and preprocess FEVER dataset"""
    try:
        # FEVER dataset is in JSONL format
        texts = []
        labels = []
        
        with open(os.path.join(data_path, 'train.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                claim = data.get('claim', '')
                label = data.get('label', 'NOT ENOUGH INFO')
                
                if claim and label in ['SUPPORTS', 'REFUTES']:
                    texts.append(claim)
                    labels.append(1 if label == 'SUPPORTS' else 0)  # Binary: SUPPORTS=1, REFUTES=0
        
        logger.info(f"Loaded FEVER dataset: {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Error loading FEVER dataset: {e}")
        return [], []

def load_fakenews_dataset(data_path):
    """Load and preprocess FakeNews dataset"""
    try:
        # Assuming CSV format with 'text' and 'label' columns
        df = pd.read_csv(os.path.join(data_path, 'fake_news.csv'))
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()  # Assuming 0=fake, 1=real
        
        # Clean data
        valid_indices = [i for i, (text, label) in enumerate(zip(texts, labels)) 
                        if pd.notna(text) and pd.notna(label)]
        texts = [texts[i] for i in valid_indices]
        labels = [int(labels[i]) for i in valid_indices]
        
        logger.info(f"Loaded FakeNews dataset: {len(texts)} samples")
        logger.info(f"Label distribution: {np.bincount(labels)}")
        
        return texts, labels
        
    except Exception as e:
        logger.error(f"Error loading FakeNews dataset: {e}")
        return [], []

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_fake_news_model(data_paths, model_name='distilbert-base-uncased', output_dir='./models/fake_news'):
    """Train a fake news detection model"""
    logger.info("Starting fake news model training...")
    
    # Load datasets
    all_texts = []
    all_labels = []
    
    for dataset_type, path in data_paths.items():
        if dataset_type == 'liar' and path:
            texts, labels = load_liar_dataset(path)
            all_texts.extend(texts)
            all_labels.extend(labels)
        elif dataset_type == 'fever' and path:
            texts, labels = load_fever_dataset(path)
            all_texts.extend(texts)
            all_labels.extend(labels)
        elif dataset_type == 'fakenews' and path:
            texts, labels = load_fakenews_dataset(path)
            all_texts.extend(texts)
            all_labels.extend(labels)
    
    if not all_texts:
        logger.error("No data loaded! Please check dataset paths.")
        return
    
    logger.info(f"Total training samples: {len(all_texts)}")
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,  # Binary classification
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = FactCheckDataset(train_texts, train_labels, tokenizer)
    val_dataset = FactCheckDataset(val_texts, val_labels, tokenizer)
    test_dataset = FactCheckDataset(test_texts, test_labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard
        save_total_limit=2,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    label_mapping = {0: 'False', 1: 'True'}
    with open(f'{output_dir}/label_mapping.json', 'w') as f:
        json.dump(label_mapping, f)
    
    logger.info(f"Model saved to {output_dir}")
    
    # Generate classification report
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    report = classification_report(
        test_labels, 
        pred_labels, 
        target_names=['False', 'True'],
        output_dict=True
    )
    
    with open(f'{output_dir}/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Training completed successfully!")
    return trainer, model, tokenizer

def create_sample_datasets():
    """Create sample datasets for testing if real datasets are not available"""
    logger.info("Creating sample datasets for testing...")
    
    # Sample fake news data
    sample_data = [
        # True statements
        ("The Earth orbits around the Sun", 1),
        ("Water boils at 100 degrees Celsius at sea level", 1),
        ("COVID-19 vaccines have been proven safe and effective", 1),
        ("Exercise can improve cardiovascular health", 1),
        ("Smoking increases the risk of lung cancer", 1),
        
        # False statements  
        ("The Earth is flat", 0),
        ("5G networks cause coronavirus", 0),
        ("Vaccines contain microchips for tracking", 0),
        ("Climate change is a hoax", 0),
        ("The moon landing was faked", 0),
        
        # Add more samples...
    ]
    
    # Expand with variations
    expanded_data = []
    for text, label in sample_data:
        expanded_data.append((text, label))
        # Add some variations
        expanded_data.append((f"It is true that {text.lower()}", label))
        expanded_data.append((f"Research shows that {text.lower()}", label))
    
    texts, labels = zip(*expanded_data)
    
    # Save as CSV for future use
    df = pd.DataFrame({'text': texts, 'label': labels})
    os.makedirs('./data/sample', exist_ok=True)
    df.to_csv('./data/sample/sample_dataset.csv', index=False)
    
    logger.info(f"Created sample dataset with {len(texts)} examples")
    return list(texts), list(labels)

def main():
    parser = argparse.ArgumentParser(description='Train fact-checking models')
    parser.add_argument('--liar_path', type=str, help='Path to LIAR dataset')
    parser.add_argument('--fever_path', type=str, help='Path to FEVER dataset') 
    parser.add_argument('--fakenews_path', type=str, help='Path to FakeNews dataset')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', 
                       help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default='./models/fake_news',
                       help='Output directory for trained model')
    parser.add_argument('--use_sample', action='store_true', 
                       help='Use sample dataset if no real datasets provided')
    
    args = parser.parse_args()
    
    # Prepare data paths
    data_paths = {
        'liar': args.liar_path,
        'fever': args.fever_path, 
        'fakenews': args.fakenews_path
    }
    
    # Check if any real datasets are provided
    has_real_data = any(path for path in data_paths.values())
    
    if not has_real_data and args.use_sample:
        logger.info("No real datasets provided, using sample data...")
        texts, labels = create_sample_datasets()
        # Use sample data for training
        data_paths = {'sample': './data/sample/sample_dataset.csv'}
    elif not has_real_data:
        logger.error("No datasets provided! Use --use_sample flag or provide dataset paths.")
        return
    
    # Train the model
    trainer, model, tokenizer = train_fake_news_model(
        data_paths=data_paths,
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main()

# Example usage:
# python train_models.py --use_sample  # Use sample data
# python train_models.py --liar_path ./data/liar --fever_path ./data/fever  # Use real datasets