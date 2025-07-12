#!/usr/bin/env python3
"""
Simplified mobile model trainer that works without TensorFlow
For Python 3.12 compatibility
"""

import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMobileModelTrainer:
    """
    Simplified trainer for mobile-optimized models without TensorFlow
    Uses scikit-learn for basic ML models
    """
    
    def __init__(self, 
                 model_dir: str = "ml_models",
                 sample_rate: int = 22050,
                 n_mels: int = 128,
                 duration: float = 2.0):
        
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.feature_size = n_mels * int(sample_rate * duration / 512)
        
        # Class labels for DCASE dataset
        self.class_labels = [
            'dog_bark', 'car_horn', 'alarm', 'glass_break', 
            'door_slam', 'siren', 'footsteps', 'speech',
            'music', 'machinery', 'nature', 'silence'
        ]
        self.num_classes = len(self.class_labels)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def extract_simple_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract simple features without librosa dependency
        """
        # Ensure audio is the right length
        target_length = int(self.sample_rate * self.duration)
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Simple FFT-based features
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Downsample to match expected feature size
        if len(magnitude) > self.feature_size:
            # Simple downsampling
            step = len(magnitude) // self.feature_size
            magnitude = magnitude[::step][:self.feature_size]
        else:
            magnitude = np.pad(magnitude, (0, self.feature_size - len(magnitude)))
        
        # Normalize
        magnitude = (magnitude - np.mean(magnitude)) / (np.std(magnitude) + 1e-8)
        
        return magnitude.astype(np.float32)
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for demonstration
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        X = np.zeros((num_samples, self.feature_size))
        y = np.random.randint(0, self.num_classes, num_samples)
        
        # Generate synthetic features for each class
        for i in range(num_samples):
            class_idx = y[i]
            
            # Create class-specific patterns
            if class_idx == 0:  # dog_bark
                pattern = np.random.normal(0, 1, self.feature_size) + np.sin(np.linspace(0, 10, self.feature_size))
            elif class_idx == 1:  # car_horn
                pattern = np.random.normal(0, 0.5, self.feature_size) + np.sin(np.linspace(0, 20, self.feature_size))
            elif class_idx == 2:  # alarm
                pattern = np.random.normal(0, 0.8, self.feature_size) + np.sin(np.linspace(0, 15, self.feature_size))
            else:
                pattern = np.random.normal(0, 1, self.feature_size)
            
            X[i] = pattern
        
        return X, y
    
    def train_simple_model(self, model_type: str = 'random_forest') -> Dict:
        """
        Train a simple ML model using scikit-learn
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import StandardScaler
            
            logger.info(f"Training {model_type} model...")
            
            # Generate training data
            X_train, y_train = self.generate_synthetic_data(2000)
            X_val, y_val = self.generate_synthetic_data(500)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Choose model
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'logistic':
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == 'svm':
                model = SVC(random_state=42, probability=True)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            logger.info("Training model...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            
            logger.info(f"Model accuracy: {accuracy:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'accuracy': accuracy,
                'model_type': model_type,
                'feature_size': self.feature_size,
                'num_classes': self.num_classes
            }
            
        except ImportError as e:
            logger.error(f"scikit-learn not available: {e}")
            logger.info("Installing scikit-learn: pip install scikit-learn")
            return None
    
    def save_model(self, model_data: Dict, output_path: str) -> bool:
        """
        Save the trained model
        """
        try:
            # Save model and scaler
            model_path = output_path.replace('.pkl', '_model.pkl')
            scaler_path = output_path.replace('.pkl', '_scaler.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(model_data['scaler'], f)
            
            # Save metadata
            metadata = {
                'model_type': model_data['model_type'],
                'accuracy': model_data['accuracy'],
                'feature_size': model_data['feature_size'],
                'num_classes': model_data['num_classes'],
                'class_labels': self.class_labels,
                'sample_rate': self.sample_rate,
                'n_mels': self.n_mels,
                'duration': self.duration,
                'created_at': datetime.now().isoformat(),
                'privacy_status': 'local_only'
            }
            
            metadata_path = output_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def benchmark_model(self, model_data: Dict, num_runs: int = 100) -> Dict:
        """
        Benchmark the model performance
        """
        try:
            import time
            
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Generate test data
            test_features = np.random.random((num_runs, self.feature_size)).astype(np.float32)
            test_features_scaled = scaler.transform(test_features)
            
            # Warm up
            for _ in range(10):
                model.predict(test_features_scaled[:1])
            
            # Benchmark
            times = []
            for i in range(num_runs):
                start_time = time.time()
                prediction = model.predict(test_features_scaled[i:i+1])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            return {
                'avg_inference_time_ms': np.mean(times),
                'std_inference_time_ms': np.std(times),
                'min_inference_time_ms': np.min(times),
                'max_inference_time_ms': np.max(times),
                'throughput_fps': 1000 / np.mean(times),
                'model_size_bytes': os.path.getsize('temp_model.pkl') if os.path.exists('temp_model.pkl') else 0,
                'privacy_status': 'local_only'
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {'error': str(e)}
    
    def train_and_export(self, 
                        model_type: str = 'random_forest',
                        epochs: int = 1) -> Dict:
        """
        Complete training and export pipeline
        """
        logger.info(f"Starting training pipeline for {model_type} model...")
        
        # Train model
        model_data = self.train_simple_model(model_type)
        
        if model_data is None:
            return {'error': 'Failed to train model'}
        
        # Save model
        output_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
        success = self.save_model(model_data, output_path)
        
        if not success:
            return {'error': 'Failed to save model'}
        
        # Benchmark model
        benchmark_results = self.benchmark_model(model_data)
        
        return {
            'success': True,
            'model_path': output_path,
            'metadata_path': output_path.replace('.pkl', '_metadata.json'),
            'benchmark_results': benchmark_results,
            'training_results': {
                'accuracy': model_data['accuracy'],
                'model_type': model_data['model_type']
            }
        }

def main():
    parser = argparse.ArgumentParser(description='Train simple mobile-optimized acoustic event detection models')
    parser.add_argument('--model-type', choices=['random_forest', 'logistic', 'svm'], default='random_forest',
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (not used for simple models)')
    parser.add_argument('--output-dir', default='ml_models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SimpleMobileModelTrainer(model_dir=args.output_dir)
    
    # Train and export model
    results = trainer.train_and_export(
        model_type=args.model_type,
        epochs=args.epochs
    )
    
    if results.get('success'):
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Model saved to: {results['model_path']}")
        print(f"Metadata saved to: {results['metadata_path']}")
        print(f"Model accuracy: {results['training_results']['accuracy']:.4f}")
        print(f"Average inference time: {results['benchmark_results']['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {results['benchmark_results']['throughput_fps']:.1f} FPS")
        print(f"Privacy status: Local processing only")
        print("="*50)
    else:
        print(f"Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 