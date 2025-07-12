import numpy as np
import json
import time
import threading
import queue
import os
import pickle
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleInferenceEngine:
    """
    Simple inference engine using scikit-learn models
    Works without TensorFlow for Python 3.12 compatibility
    """
    
    def __init__(self, enable_optimization: bool = True):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.model_loaded = False
        self.enable_optimization = enable_optimization
        self.model_size_bytes = 0
        self.inference_count = 0
        self.total_inference_time = 0
        
    def load_model(self, model_path: str) -> bool:
        """
        Load scikit-learn model completely offline
        """
        try:
            # Verify files exist locally
            model_file = model_path.replace('.pkl', '_model.pkl')
            scaler_file = model_path.replace('.pkl', '_scaler.pkl')
            metadata_file = model_path.replace('.pkl', '_metadata.json')
            
            if not all(os.path.exists(f) for f in [model_file, scaler_file, metadata_file]):
                logger.error(f"Model files not found: {model_path}")
                return False
            
            # Load model
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Get model size
            self.model_size_bytes = sum(os.path.getsize(f) for f in [model_file, scaler_file, metadata_file])
            logger.info(f"Model loaded successfully: {model_path} ({self.model_size_bytes} bytes)")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """
        Run inference locally without any data transmission
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'privacy_status': 'local_only',
                'success': False
            }
        
        try:
            start_time = time.time()
            
            # Ensure input data is the right shape
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            
            # Scale features
            input_scaled = self.scaler.transform(input_data)
            
            # Run inference (completely local)
            predictions = self.model.predict_proba(input_scaled)[0]
            predicted_class = self.model.predict(input_scaled)[0]
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            # Get class label
            class_labels = self.metadata.get('class_labels', [])
            predicted_label = class_labels[predicted_class] if predicted_class < len(class_labels) else 'unknown'
            
            return {
                'predictions': predictions.tolist(),
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': float(np.max(predictions)),
                'inference_time_ms': inference_time,
                'privacy_status': 'local_only',
                'data_transmitted': False,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'error': str(e),
                'privacy_status': 'local_only',
                'success': False
            }
    
    def get_model_info(self) -> Dict:
        """
        Get model information without external calls
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'model_type': self.metadata.get('model_type', 'unknown'),
            'feature_size': self.metadata.get('feature_size', 0),
            'num_classes': self.metadata.get('num_classes', 0),
            'accuracy': self.metadata.get('accuracy', 0),
            'model_size_bytes': self.model_size_bytes,
            'inference_count': self.inference_count,
            'avg_inference_time_ms': self.total_inference_time / max(self.inference_count, 1),
            'optimization_enabled': self.enable_optimization,
            'privacy_status': 'local_only',
            'class_labels': self.metadata.get('class_labels', [])
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
        self.scaler = None
        self.metadata = None
        self.model_loaded = False
        logger.info("Simple inference engine cleaned up")

class SimpleAudioFeatureExtractor:
    """
    Simple audio feature extraction without external dependencies
    """
    
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128, duration: float = 2.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        self.feature_size = n_mels * int(sample_rate * duration / 512)
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract simple FFT-based features
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
            step = len(magnitude) // self.feature_size
            magnitude = magnitude[::step][:self.feature_size]
        else:
            magnitude = np.pad(magnitude, (0, self.feature_size - len(magnitude)))
        
        # Normalize
        magnitude = (magnitude - np.mean(magnitude)) / (np.std(magnitude) + 1e-8)
        
        return magnitude.astype(np.float32)

class SimpleAudioProcessor:
    """
    Real-time audio processor using simple inference engine
    """
    
    def __init__(self, 
                 inference_engine: SimpleInferenceEngine,
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_duration: float = 2.0,
                 confidence_threshold: float = 0.6):
        
        self.inference_engine = inference_engine
        self.feature_extractor = SimpleAudioFeatureExtractor(sample_rate)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.confidence_threshold = confidence_threshold
        
        # Audio buffer
        self.audio_buffer = np.array([])
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=50)
        
        # Privacy metrics
        self.total_audio_processed = 0
        self.events_detected = 0
        self.privacy_violations = 0
        
    def start_processing(self):
        """Start real-time processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Simple audio processing started")
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        logger.info("Audio processing stopped")
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process audio chunk locally without any data transmission
        """
        try:
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            self.total_audio_processed += len(audio_chunk)
            
            # Check if we have enough data
            if len(self.audio_buffer) >= self.buffer_size:
                # Extract segment for processing
                segment = self.audio_buffer[:self.buffer_size]
                
                # Check for sufficient energy (not silence)
                if self._has_sufficient_energy(segment):
                    # Extract features locally
                    features = self.feature_extractor.extract_features(segment)
                    
                    # Run inference locally
                    result = self.inference_engine.predict(features)
                    
                    if result['success'] and self._is_valid_prediction(result):
                        # Update metrics
                        self.events_detected += 1
                        
                        # Remove processed samples with overlap
                        self.audio_buffer = self.audio_buffer[self.buffer_size//2:]
                        
                        return self._format_result(result, segment)
                
                # Remove old samples to prevent buffer overflow
                self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def _has_sufficient_energy(self, audio: np.ndarray) -> bool:
        """Check if audio has sufficient energy for processing"""
        rms = np.sqrt(np.mean(audio**2))
        return rms > 0.01  # Threshold for non-silence
    
    def _is_valid_prediction(self, result: Dict) -> bool:
        """Validate prediction results"""
        return result.get('confidence', 0) > self.confidence_threshold
    
    def _format_result(self, result: Dict, audio_segment: np.ndarray) -> Dict:
        """Format inference result with privacy information"""
        return {
            'event_type': result.get('predicted_label', 'unknown'),
            'confidence': result.get('confidence', 0),
            'inference_time_ms': result.get('inference_time_ms', 0),
            'privacy_status': 'local_only',
            'data_transmitted': False,
            'timestamp': datetime.now().isoformat(),
            'audio_duration_ms': len(audio_segment) / self.sample_rate * 1000,
            'all_predictions': dict(zip(
                result.get('class_labels', []),
                result.get('predictions', [])
            ))
        }
    
    def _processing_loop(self):
        """Background processing loop"""
        while self.is_processing:
            try:
                # Process any queued audio
                while not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get_nowait()
                    result = self.process_audio_chunk(audio_chunk)
                    if result:
                        self.result_queue.put(result)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def get_privacy_metrics(self) -> Dict:
        """Get privacy and processing metrics"""
        return {
            'total_audio_processed_bytes': self.total_audio_processed * 4,  # Assuming float32
            'events_detected': self.events_detected,
            'privacy_violations': self.privacy_violations,
            'processing_active': self.is_processing,
            'buffer_size_bytes': len(self.audio_buffer) * 4,
            'privacy_status': 'local_only',
            'data_transmitted': False
        }
    
    def reset_metrics(self):
        """Reset processing metrics"""
        self.total_audio_processed = 0
        self.events_detected = 0
        self.privacy_violations = 0

# Factory function for creating inference engines
def create_simple_inference_engine(**kwargs) -> SimpleInferenceEngine:
    """
    Factory function to create simple inference engines
    """
    return SimpleInferenceEngine(**kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Example of setting up simple inference
    engine = create_simple_inference_engine(enable_optimization=True)
    
    # Load model (replace with actual model path)
    model_loaded = engine.load_model("ml_models/random_forest_model.pkl")
    
    if model_loaded:
        # Create audio processor
        processor = SimpleAudioProcessor(engine)
        
        # Start processing
        processor.start_processing()
        
        # Simulate audio input
        sample_audio = np.random.random(22050 * 2)  # 2 seconds of audio
        
        # Process audio
        result = processor.process_audio_chunk(sample_audio)
        
        if result:
            print(f"Detected event: {result['event_type']} (confidence: {result['confidence']:.3f})")
            print(f"Privacy status: {result['privacy_status']}")
        
        # Get metrics
        metrics = processor.get_privacy_metrics()
        print(f"Privacy metrics: {metrics}")
        
        # Cleanup
        processor.stop_processing()
        engine.cleanup() 