"""
Comprehensive Model Evaluation and Benchmarking for TruthMate
Compare different models and provide detailed performance analysis
"""
import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import time
import requests
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.test_data = []
        
    def load_test_datasets(self, dataset_paths: Dict[str, str]) -> List[Dict]:
        """Load comprehensive test datasets"""
        test_samples = []
        
        # Load various test sets
        for dataset_name, path in dataset_paths.items():
            logger.info(f"Loading test data from {dataset_name}")
            
            try:
                if dataset_name == 'climate_facts':
                    samples = self.load_climate_test_data(path)
                elif dataset_name == 'health_facts':
                    samples = self.load_health_test_data(path)
                elif dataset_name == 'political_facts':
                    samples = self.load_political_test_data(path)
                elif dataset_name == 'custom_test':
                    samples = self.load_custom_test_data(path)
                else:
                    samples = self.load_generic_test_data(path)
                
                for sample in samples:
                    sample['source_dataset'] = dataset_name
                    
                test_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} test samples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")
        
        # Add curated test cases
        test_samples.extend(self.get_curated_test_cases())
        
        logger.info(f"Total test samples: {len(test_samples)}")
        return test_samples
    
    def get_curated_test_cases(self) -> List[Dict]:
        """High-quality curated test cases across different domains"""
        return [
            # Medical/Health Facts
            {
                'claim': 'COVID-19 vaccines have been proven safe and effective in large clinical trials.',
                'evidence': 'Multiple Phase 3 clinical trials with tens of thousands of participants showed 90%+ efficacy and acceptable safety profiles.',
                'true_label': 2,  # True
                'category': 'health',
                'difficulty': 'easy',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Drinking bleach can cure coronavirus infections.',
                'evidence': 'No scientific evidence supports this claim. Bleach is toxic and can cause severe internal damage.',
                'true_label': 0,  # False
                'category': 'health',
                'difficulty': 'easy',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Natural immunity from COVID-19 infection is always better than vaccine immunity.',
                'evidence': 'Studies show variable and waning natural immunity. Vaccination provides more consistent protection.',
                'true_label': 1,  # Misleading
                'category': 'health',
                'difficulty': 'hard',
                'source_dataset': 'curated'
            },
            
            # Climate Science
            {
                'claim': 'Human activities are the primary driver of current climate change.',
                'evidence': 'Overwhelming scientific consensus based on multiple lines of evidence including temperature records, ice cores, and climate models.',
                'true_label': 2,  # True
                'category': 'climate',
                'difficulty': 'easy',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Climate has always changed naturally, so current warming is natural.',
                'evidence': 'While climate has changed naturally in the past, current rapid warming correlates with industrial emissions and cannot be explained by natural factors alone.',
                'true_label': 1,  # Misleading
                'category': 'climate',
                'difficulty': 'hard',
                'source_dataset': 'curated'
            },
            {
                'claim': 'CO2 is plant food, so more CO2 is always beneficial.',
                'evidence': 'While plants use CO2, higher concentrations also cause warming, extreme weather, and ocean acidification that can harm ecosystems.',
                'true_label': 1,  # Misleading
                'category': 'climate',
                'difficulty': 'medium',
                'source_dataset': 'curated'
            },
            
            # Technology/Science
            {
                'claim': '5G networks cause cancer and coronavirus.',
                'evidence': 'No scientific evidence links 5G to cancer or viruses. Radio waves used in 5G are non-ionizing and cannot damage DNA.',
                'true_label': 0,  # False
                'category': 'technology',
                'difficulty': 'easy',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Artificial intelligence will replace all human jobs within 10 years.',
                'evidence': 'While AI will automate some jobs, experts predict gradual change with new job creation. Complete replacement unlikely in 10 years.',
                'true_label': 1,  # Misleading
                'category': 'technology',
                'difficulty': 'medium',
                'source_dataset': 'curated'
            },
            
            # Social/Political
            {
                'claim': 'Voter fraud determined the outcome of the 2020 US presidential election.',
                'evidence': 'Multiple audits, recounts, and court cases found no evidence of fraud sufficient to change the election outcome.',
                'true_label': 0,  # False
                'category': 'politics',
                'difficulty': 'medium',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Social media platforms have some bias in content moderation.',
                'evidence': 'Studies and reports indicate various biases in content moderation, though the extent and direction are debated.',
                'true_label': 2,  # True (but nuanced)
                'category': 'politics',
                'difficulty': 'hard',
                'source_dataset': 'curated'
            },
            
            # Economics
            {
                'claim': 'Minimum wage increases always reduce employment.',
                'evidence': 'Economic research shows mixed results. Some studies find small employment effects, others find minimal impact.',
                'true_label': 1,  # Misleading (too absolute)
                'category': 'economics',
                'difficulty': 'hard',
                'source_dataset': 'curated'
            },
            {
                'claim': 'Cryptocurrency has no environmental impact.',
                'evidence': 'Bitcoin and proof-of-work cryptocurrencies consume significant electricity. Some newer cryptocurrencies use more efficient methods.',
                'true_label': 0,  # False
                'category': 'economics',
                'difficulty': 'medium',
                'source_dataset': 'curated'
            },
            
            # Edge Cases - Tricky Examples
            {
                'claim': 'Water is wet.',
                'evidence': 'This is a philosophical question. Water makes things wet, but whether water itself is wet is debated.',
                'true_label': 1,  # Misleading (depends on definition)
                'category': 'edge_case',
                'difficulty': 'hard',
                'source_dataset': 'curated'
            },
            {
                'claim': 'All swans are white.',
                'evidence': 'Black swans exist in Australia and other regions. This claim is factually incorrect.',
                'true_label': 0,  # False
                'category': 'edge_case',
                'difficulty': 'easy',
                'source_dataset': 'curated'
            },
            
            # Temporal/Context Dependent
            {
                'claim': 'Masks are effective at preventing respiratory disease transmission.',
                'evidence': 'Multiple studies during COVID-19 pandemic confirmed mask effectiveness for reducing transmission of respiratory droplets.',
                'true_label': 2,  # True
                'category': 'health',
                'difficulty': 'medium',
                'source_dataset': 'curated'
            }
        ]
    
    def load_climate_test_data(self, path: str) -> List[Dict]:
        """Load climate-specific test cases"""
        # Implementation for climate fact dataset
        return []
    
    def load_health_test_data(self, path: str) -> List[Dict]:
        """Load health-specific test cases"""
        return []
    
    def load_political_test_data(self, path: str) -> List[Dict]:
        """Load political fact-checking test cases"""
        return []
    
    def load_custom_test_data(self, path: str) -> List[Dict]:
        """Load custom test dataset"""
        try:
            df = pd.read_csv(path)
            samples = []
            
            for _, row in df.iterrows():
                samples.append({
                    'claim': row.get('claim', row.get('text', '')),
                    'evidence': row.get('evidence', ''),
                    'true_label': int(row.get('label', row.get('true_label', 1))),
                    'category': row.get('category', 'general'),
                    'difficulty': row.get('difficulty', 'medium')
                })
            
            return samples
        except Exception as e:
            logger.error(f"Error loading custom test data: {e}")
            return []
    
    def load_generic_test_data(self, path: str) -> List[Dict]:
        """Load generic test data format"""
        return self.load_custom_test_data(path)
    
    def evaluate_transformer_model(self, model_path: str, test_data: List[Dict]) -> Dict:
        """Evaluate a trained transformer model"""
        logger.info(f"Evaluating transformer model: {model_path}")
        
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Create pipeline
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            predictions = []
            true_labels = []
            inference_times = []
            confidence_scores = []
            
            for sample in test_data:
                start_time = time.time()
                
                # Prepare input text
                text = sample['claim']
                if sample.get('evidence'):
                    text += f" [Evidence: {sample['evidence']}]"
                
                # Get prediction
                result = classifier(text)
                
                # Extract prediction (assuming labels are LABEL_0, LABEL_1, LABEL_2)
                label_mapping = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
                pred_label = label_mapping.get(result[0]['label'], 1)
                confidence = result[0]['score']
                
                predictions.append(pred_label)
                true_labels.append(sample['true_label'])
                confidence_scores.append(confidence)
                
                inference_times.append(time.time() - start_time)
            
            # Calculate metrics
            metrics = self.calculate_detailed_metrics(
                true_labels, predictions, confidence_scores
            )
            
            metrics.update({
                'avg_inference_time': np.mean(inference_times),
                'total_samples': len(test_data),
                'model_type': 'transformer',
                'model_path': model_path
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating transformer model: {e}")
            return {'error': str(e)}
    
    def evaluate_api_service(self, api_url: str, test_data: List[Dict]) -> Dict:
        """Evaluate API-based fact-checking service"""
        logger.info(f"Evaluating API service: {api_url}")
        
        predictions = []
        true_labels = []
        inference_times = []
        confidence_scores = []
        errors = 0
        
        for sample in test_data:
            try:
                start_time = time.time()
                
                # Make API request
                payload = {
                    'claim': sample['claim'],
                    'evidence': sample.get('evidence', ''),
                    'analyze_sources': True,
                    'get_explanation': True
                }
                
                response = requests.post(
                    f"{api_url}/verify",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract prediction
                    verdict = result.get('verdict', 'Unknown')
                    verdict_mapping = {'True': 2, 'False': 0, 'Misleading': 1, 'Unknown': 1}
                    pred_label = verdict_mapping.get(verdict, 1)
                    
                    confidence = result.get('confidence_score', 0.5)
                    
                    predictions.append(pred_label)
                    confidence_scores.append(confidence)
                else:
                    # Handle API errors
                    predictions.append(1)  # Default to "Unknown"
                    confidence_scores.append(0.5)
                    errors += 1
                
                true_labels.append(sample['true_label'])
                inference_times.append(time.time() - start_time)
                
            except Exception as e:
                logger.error(f"API request failed: {e}")
                predictions.append(1)
                confidence_scores.append(0.5)
                true_labels.append(sample['true_label'])
                inference_times.append(30.0)  # Timeout
                errors += 1
        
        # Calculate metrics
        metrics = self.calculate_detailed_metrics(
            true_labels, predictions, confidence_scores
        )
        
        metrics.update({
            'avg_inference_time': np.mean(inference_times),
            'total_samples': len(test_data),
            'api_errors': errors,
            'error_rate': errors / len(test_data),
            'model_type': 'api_service',
            'api_url': api_url
        })
        
        return metrics
    
    def calculate_detailed_metrics(self, true_labels: List[int], 
                                 predictions: List[int], 
                                 confidence_scores: List[float]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic classification metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2])
        
        # Calculate accuracy by category if available
        category_metrics = {}
        
        # Confidence calibration
        confidence_buckets = np.linspace(0, 1, 11)
        calibration_accuracy = []
        calibration_confidence = []
        
        for i in range(len(confidence_buckets) - 1):
            lower, upper = confidence_buckets[i], confidence_buckets[i + 1]
            mask = (np.array(confidence_scores) >= lower) & (np.array(confidence_scores) < upper)
            
            if np.sum(mask) > 0:
                bucket_accuracy = accuracy_score(
                    np.array(true_labels)[mask],
                    np.array(predictions)[mask]
                )
                bucket_confidence = np.mean(np.array(confidence_scores)[mask])
                
                calibration_accuracy.append(bucket_accuracy)
                calibration_confidence.append(bucket_confidence)
        
        # Expected Calibration Error (ECE)
        ece = 0
        for acc, conf in zip(calibration_accuracy, calibration_confidence):
            ece += abs(acc - conf)
        ece /= len(calibration_accuracy) if calibration_accuracy else 1
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_false': precision_per_class[0] if len(precision_per_class) > 0 else 0,
            'precision_misleading': precision_per_class[1] if len(precision_per_class) > 1 else 0,
            'precision_true': precision_per_class[2] if len(precision_per_class) > 2 else 0,
            'recall_false': recall_per_class[0] if len(recall_per_class) > 0 else 0,
            'recall_misleading': recall_per_class[1] if len(recall_per_class) > 1 else 0,
            'recall_true': recall_per_class[2] if len(recall_per_class) > 2 else 0,
            'f1_false': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_misleading': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'f1_true': f1_per_class[2] if len(f1_per_class) > 2 else 0,
            'confusion_matrix': cm.tolist(),
            'expected_calibration_error': ece,
            'avg_confidence': np.mean(confidence_scores),
            'confidence_std': np.std(confidence_scores),
        }
    
    def run_comprehensive_benchmark(self, models_config: Dict, test_datasets: Dict, 
                                  output_dir: str = './benchmark_results'):
        """Run comprehensive benchmark across multiple models"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_data = self.load_test_datasets(test_datasets)
        if not test_data:
            logger.error("No test data loaded!")
            return
        
        # Store test data
        with open(f'{output_dir}/test_data.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        
        benchmark_results = {}
        
        # Evaluate each model
        for model_name, model_config in models_config.items():
            logger.info(f"Benchmarking {model_name}")
            
            if model_config['type'] == 'transformer':
                results = self.evaluate_transformer_model(
                    model_config['path'], test_data
                )
            elif model_config['type'] == 'api':
                results = self.evaluate_api_service(
                    model_config['url'], test_data
                )
            else:
                logger.warning(f"Unknown model type: {model_config['type']}")
                continue
            
            benchmark_results[model_name] = results
            
            # Save intermediate results
            with open(f'{output_dir}/results_{model_name}.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        # Generate comparison report
        self.generate_comparison_report(benchmark_results, output_dir)
        
        # Save complete results
        with open(f'{output_dir}/complete_benchmark.json', 'w') as f:
            json.dump({
                'benchmark_results': benchmark_results,
                'test_data_summary': {
                    'total_samples': len(test_data),
                    'label_distribution': dict(pd.Series([s['true_label'] for s in test_data]).value_counts()),
                    'category_distribution': dict(pd.Series([s.get('category', 'unknown') for s in test_data]).value_counts()),
                    'difficulty_distribution': dict(pd.Series([s.get('difficulty', 'unknown') for s in test_data]).value_counts())
                }
            }, f, indent=2)
        
        logger.info(f"Comprehensive benchmark completed! Results saved to {output_dir}")
        
        return benchmark_results
    
    def generate_comparison_report(self, results: Dict, output_dir: str):
        """Generate detailed comparison report with visualizations"""
        
        # Create comparison DataFrame
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'avg_inference_time']
        comparison_data = []
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                row = {'Model': model_name}
                for metric in metrics:
                    row[metric] = model_results.get(metric, 0)
                row['Total Samples'] = model_results.get('total_samples', 0)
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        
        # Generate visualizations
        self.create_benchmark_visualizations(results, output_dir)
        
        # Generate summary report
        with open(f'{output_dir}/benchmark_summary.md', 'w') as f:
            f.write("# TruthMate Model Benchmark Results\n\n")
            f.write(f"**Benchmark Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # Best performing models
            if not df.empty:
                best_accuracy = df.loc[df['accuracy'].idxmax(), 'Model']
                best_f1 = df.loc[df['f1_score'].idxmax(), 'Model']
                fastest = df.loc[df['avg_inference_time'].idxmin(), 'Model']
                
                f.write("## Key Findings\n\n")
                f.write(f"- **Best Accuracy:** {best_accuracy} ({df['accuracy'].max():.3f})\n")
                f.write(f"- **Best F1 Score:** {best_f1} ({df['f1_score'].max():.3f})\n")
                f.write(f"- **Fastest Model:** {fastest} ({df['avg_inference_time'].min():.3f}s per sample)\n\n")
            
            f.write("## Detailed Analysis\n\n")
            for model_name, model_results in results.items():
                f.write(f"### {model_name}\n")
                if 'error' in model_results:
                    f.write(f"**Error:** {model_results['error']}\n\n")
                else:
                    f.write(f"- **Type:** {model_results.get('model_type', 'Unknown')}\n")
                    f.write(f"- **Accuracy:** {model_results.get('accuracy', 0):.3f}\n")
                    f.write(f"- **F1 Score:** {model_results.get('f1_score', 0):.3f}\n")
                    f.write(f"- **Inference Time:** {model_results.get('avg_inference_time', 0):.3f}s\n")
                    
                    if model_results.get('api_errors', 0) > 0:
                        f.write(f"- **API Errors:** {model_results['api_errors']} ({model_results.get('error_rate', 0):.2%})\n")
                    
                    f.write("\n")
        
        logger.info("Comparison report generated!")
    
    def create_benchmark_visualizations(self, results: Dict, output_dir: str):
        """Create comprehensive benchmark visualizations"""
        
        # Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = []
        accuracies = []
        f1_scores = []
        inference_times = []
        
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                models.append(model_name)
                accuracies.append(model_results.get('accuracy', 0))
                f1_scores.append(model_results.get('f1_score', 0))
                inference_times.append(model_results.get('avg_inference_time', 0))
        
        # Accuracy comparison
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        axes[0, 1].bar(models, f1_scores, color='lightgreen')
        axes[0, 1].set_title('Model F1 Score Comparison')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        axes[1, 0].bar(models, inference_times, color='orange')
        axes[1, 0].set_title('Average Inference Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Performance vs Speed scatter
        axes[1, 1].scatter(inference_times, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (inference_times[i], f1_scores[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Inference Time (seconds)')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Performance vs Speed Trade-off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class performance heatmap
        if len(results) > 0:
            class_metrics = ['f1_false', 'f1_misleading', 'f1_true']
            heatmap_data = []
            
            for model_name, model_results in results.items():
                if 'error' not in model_results:
                    row = [model_results.get(metric, 0) for metric in class_metrics]
                    heatmap_data.append(row)
            
            if heatmap_data:
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    heatmap_data,
                    xticklabels=['False', 'Misleading', 'True'],
                    yticklabels=[m for m in results.keys() if 'error' not in results[m]],
                    annot=True,
                    cmap='YlOrRd',
                    fmt='.3f'
                )
                plt.title('Per-Class F1 Scores')
                plt.ylabel('Models')
                plt.xlabel('Classes')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/per_class_performance.png', dpi=300, bbox_inches='tight')
                plt.close()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Benchmarking')
    
    parser.add_argument('--config', type=str, required=True,
                       help='JSON config file with models and datasets')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Output directory for benchmark results')
    parser.add_argument('--test_api', type=str, 
                       help='Test a specific API endpoint')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with curated examples only')
    
    args = parser.parse_args()
    
    benchmark = ModelBenchmark()
    
    if args.quick_test:
        # Quick test with curated examples
        test_data = benchmark.get_curated_test_cases()
        
        if args.test_api:
            logger.info(f"Quick testing API: {args.test_api}")
            results = benchmark.evaluate_api_service(args.test_api, test_data)
            print(json.dumps(results, indent=2))
        else:
            logger.info("Use --test_api to specify an API endpoint for quick testing")
        
        return
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        models_config = config.get('models', {})
        test_datasets = config.get('test_datasets', {})
        
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark(
            models_config, test_datasets, args.output_dir
        )
        
        logger.info("Benchmarking completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")

if __name__ == "__main__":
    main()

# Example usage:
# python benchmark.py --quick_test --test_api http://localhost:5000
# python benchmark.py --config benchmark_config.json --output_dir ./results