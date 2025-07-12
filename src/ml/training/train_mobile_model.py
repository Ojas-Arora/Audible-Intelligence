import tensorflow as tf
import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileModelTrainer:
    """
    Trainer for mobile-optimized TensorFlow Lite models with privacy considerations
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
        self.input_shape = (1, n_mels, int(sample_rate * duration / 512), 1)
        
        # Class labels for DCASE dataset
        self.class_labels = [
            'dog_bark', 'car_horn', 'alarm', 'glass_break', 
            'door_slam', 'siren', 'footsteps', 'speech',
            'music', 'machinery', 'nature', 'silence'
        ]
        self.num_classes = len(self.class_labels)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def create_mobile_model(self) -> tf.keras.Model:
        """
        Create a mobile-optimized CNN model for acoustic event detection
        """
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape[1:]),
            
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global average pooling
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            
            # Output layer
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lightweight_model(self) -> tf.keras.Model:
        """
        Create an ultra-lightweight model for edge devices
        """
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.input_shape[1:]),
            
            # Lightweight convolutional blocks
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            # Minimal dense layers
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for demonstration
        In production, this would be replaced with real DCASE dataset
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        X = np.random.random((num_samples,) + self.input_shape[1:]).astype(np.float32)
        y = np.random.randint(0, self.num_classes, num_samples)
        y_onehot = tf.keras.utils.to_categorical(y, self.num_classes)
        
        return X, y_onehot
    
    def train_model(self, 
                   model: tf.keras.Model, 
                   epochs: int = 50,
                   batch_size: int = 32,
                   validation_split: float = 0.2) -> tf.keras.callbacks.History:
        """
        Train the model with synthetic data
        """
        logger.info("Generating training data...")
        X_train, y_train = self.generate_synthetic_data(2000)
        X_val, y_val = self.generate_synthetic_data(500)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        logger.info("Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, 
                         model: tf.keras.Model, 
                         output_path: str,
                         quantization_type: str = 'dynamic') -> bool:
        """
        Convert Keras model to TensorFlow Lite with optimization
        """
        try:
            logger.info(f"Converting model to TensorFlow Lite with {quantization_type} quantization...")
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Configure optimization
            if quantization_type == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif quantization_type == 'float16':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Get model size
            model_size = os.path.getsize(output_path)
            logger.info(f"Model saved to {output_path} ({model_size} bytes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            return False
    
    def _representative_dataset(self):
        """
        Representative dataset for quantization
        """
        for _ in range(100):
            yield [np.random.random(self.input_shape).astype(np.float32)]
    
    def benchmark_model(self, model_path: str, num_runs: int = 100) -> Dict:
        """
        Benchmark the TensorFlow Lite model performance
        """
        try:
            import tensorflow as tf
            
            # Load model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            # Get input details
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            
            # Generate test data
            test_input = np.random.random(input_shape).astype(np.float32)
            
            # Warm up
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            return {
                'avg_inference_time_ms': np.mean(times),
                'std_inference_time_ms': np.std(times),
                'min_inference_time_ms': np.min(times),
                'max_inference_time_ms': np.max(times),
                'throughput_fps': 1000 / np.mean(times),
                'model_size_bytes': os.path.getsize(model_path),
                'input_shape': input_shape.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {'error': str(e)}
    
    def save_model_metadata(self, model_path: str, metadata: Dict):
        """
        Save model metadata for the mobile app
        """
        metadata_path = model_path.replace('.tflite', '_metadata.json')
        
        metadata.update({
            'class_labels': self.class_labels,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'duration': self.duration,
            'input_shape': self.input_shape,
            'created_at': datetime.now().isoformat(),
            'privacy_status': 'local_only'
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
    
    def train_and_export(self, 
                        model_type: str = 'standard',
                        quantization: str = 'dynamic',
                        epochs: int = 50) -> Dict:
        """
        Complete training and export pipeline
        """
        logger.info(f"Starting training pipeline for {model_type} model...")
        
        # Create model
        if model_type == 'lightweight':
            model = self.create_lightweight_model()
        else:
            model = self.create_mobile_model()
        
        # Train model
        history = self.train_model(model, epochs=epochs)
        
        # Save Keras model
        keras_path = os.path.join(self.model_dir, f"{model_type}_model.h5")
        model.save(keras_path)
        logger.info(f"Keras model saved to {keras_path}")
        
        # Convert to TensorFlow Lite
        tflite_path = os.path.join(self.model_dir, f"{model_type}_model_{quantization}.tflite")
        success = self.convert_to_tflite(model, tflite_path, quantization)
        
        if not success:
            return {'error': 'Failed to convert model to TensorFlow Lite'}
        
        # Benchmark model
        benchmark_results = self.benchmark_model(tflite_path)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'quantization': quantization,
            'training_history': {
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1])
            },
            'benchmark_results': benchmark_results
        }
        
        self.save_model_metadata(tflite_path, metadata)
        
        return {
            'success': True,
            'model_path': tflite_path,
            'metadata_path': tflite_path.replace('.tflite', '_metadata.json'),
            'benchmark_results': benchmark_results,
            'training_history': metadata['training_history']
        }

def main():
    parser = argparse.ArgumentParser(description='Train mobile-optimized acoustic event detection models')
    parser.add_argument('--model-type', choices=['standard', 'lightweight'], default='standard',
                       help='Type of model to train')
    parser.add_argument('--quantization', choices=['none', 'dynamic', 'float16', 'int8'], default='dynamic',
                       help='Quantization type for TensorFlow Lite model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--output-dir', default='ml_models', help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MobileModelTrainer(model_dir=args.output_dir)
    
    # Train and export model
    results = trainer.train_and_export(
        model_type=args.model_type,
        quantization=args.quantization,
        epochs=args.epochs
    )
    
    if results.get('success'):
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Model saved to: {results['model_path']}")
        print(f"Metadata saved to: {results['metadata_path']}")
        print(f"Model size: {results['benchmark_results']['model_size_bytes']} bytes")
        print(f"Average inference time: {results['benchmark_results']['avg_inference_time_ms']:.2f} ms")
        print(f"Throughput: {results['benchmark_results']['throughput_fps']:.1f} FPS")
        print(f"Final accuracy: {results['training_history']['final_accuracy']:.4f}")
        print(f"Privacy status: Local processing only")
        print("="*50)
    else:
        print(f"Training failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 