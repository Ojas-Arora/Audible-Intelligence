import numpy as np
import json
import time
import threading
import queue
import os
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyPreservingInferenceEngine(ABC):
    """
    Abstract base class for privacy-preserving on-device inference engines
    """
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load model without any network calls"""
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict:
        """Run inference locally without data transmission"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """Get model information without external calls"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass

class TensorFlowLiteEngine(PrivacyPreservingInferenceEngine):
    """
    TensorFlow Lite inference engine optimized for privacy and mobile deployment
    """
    
    def __init__(self, enable_quantization: bool = True):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        self.enable_quantization = enable_quantization
        self.model_size_bytes = 0
        self.inference_count = 0
        self.total_inference_time = 0
        
    def load_model(self, model_path: str) -> bool:
        """
        Load TensorFlow Lite model completely offline
        """
        try:
            import tensorflow as tf
            
            # Verify file exists locally
            if not os.path.exists(model_path):
                logger.error(f"Model file not found locally: {model_path}")
                return False
            
            # Get model size for privacy monitoring
            self.model_size_bytes = os.path.getsize(model_path)
            logger.info(f"Loading local model: {model_path} ({self.model_size_bytes} bytes)")
            
            # Load model with privacy-focused configuration
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=1  # Single thread for deterministic behavior
            )
            
            # Allocate tensors
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_loaded = True
            logger.info("TensorFlow Lite model loaded successfully (offline)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TensorFlow Lite model: {e}")
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
            
            # Ensure input data is local and not transmitted
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference (completely local)
            self.interpreter.invoke()
            
            # Get output (local only)
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return {
                'predictions': output_data.tolist(),
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
            'input_shape': self.input_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'output_dtype': str(self.output_details[0]['dtype']),
            'model_size_bytes': self.model_size_bytes,
            'inference_count': self.inference_count,
            'avg_inference_time_ms': self.total_inference_time / max(self.inference_count, 1),
            'quantization_enabled': self.enable_quantization,
            'privacy_status': 'local_only'
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.interpreter = None
        self.model_loaded = False
        logger.info("TensorFlow Lite engine cleaned up")

class AudioFeatureExtractor:
    """
    Local audio feature extraction without cloud dependencies
    """
    
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features locally
        """
        try:
            import librosa
            
            # Ensure audio is the right length
            target_length = int(self.sample_rate * 2.0)  # 2 seconds
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize locally
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
            
            # Reshape for model input
            return log_mel_spec.reshape(1, self.n_mels, -1, 1).astype(np.float32)
            
        except ImportError:
            logger.warning("librosa not available, using simple features")
            return self._simple_feature_extraction(audio)
    
    def _simple_feature_extraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple feature extraction without external dependencies
        """
        # Simple FFT-based features
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Downsample to match expected input size
        target_size = self.n_mels * 128  # Approximate size
        if len(magnitude) > target_size:
            magnitude = magnitude[:target_size]
        else:
            magnitude = np.pad(magnitude, (0, target_size - len(magnitude)))
        
        # Normalize
        magnitude = (magnitude - np.mean(magnitude)) / (np.std(magnitude) + 1e-8)
        
        return magnitude.reshape(1, self.n_mels, -1, 1).astype(np.float32)

class PrivacyPreservingAudioProcessor:
    """
    Real-time audio processor that ensures all processing stays local
    """
    
    def __init__(self, 
                 inference_engine: PrivacyPreservingInferenceEngine,
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_duration: float = 2.0,
                 confidence_threshold: float = 0.6):
        
        self.inference_engine = inference_engine
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
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
        logger.info("Privacy-preserving audio processing started")
    
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
                    features = self.feature_extractor.extract_mel_spectrogram(segment)
                    
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
        if 'predictions' not in result:
            return False
        
        predictions = result['predictions']
        if isinstance(predictions, list) and len(predictions) > 0:
            max_confidence = max(predictions[0]) if isinstance(predictions[0], list) else max(predictions)
            return max_confidence > self.confidence_threshold
        
        return False
    
    def _format_result(self, result: Dict, audio_segment: np.ndarray) -> Dict:
        """Format inference result with privacy information"""
        predictions = result['predictions']
        
        # Get top prediction
        if isinstance(predictions, list) and len(predictions) > 0:
            pred_array = predictions[0] if isinstance(predictions[0], list) else predictions
            top_class_idx = np.argmax(pred_array)
            confidence = float(pred_array[top_class_idx])
            
            # Class labels (local only)
            class_labels = [
                'dog_bark', 'car_horn', 'alarm', 'glass_break', 
                'door_slam', 'siren', 'footsteps', 'speech',
                'music', 'machinery', 'nature', 'silence'
            ]
            
            predicted_class = class_labels[top_class_idx] if top_class_idx < len(class_labels) else 'unknown'
            
            return {
                'event_type': predicted_class,
                'confidence': confidence,
                'inference_time_ms': result.get('inference_time_ms', 0),
                'privacy_status': 'local_only',
                'data_transmitted': False,
                'timestamp': datetime.now().isoformat(),
                'audio_duration_ms': len(audio_segment) / self.sample_rate * 1000,
                'all_predictions': {
                    label: float(pred) for label, pred in zip(class_labels, pred_array)
                }
            }
        
        return result
    
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

class ModelOptimizer:
    """
    Model optimization utilities for mobile deployment
    """
    
    @staticmethod
    def quantize_model(model_path: str, output_path: str, quantization_type: str = 'dynamic') -> bool:
        """
        Quantize TensorFlow model for mobile deployment
        """
        try:
            import tensorflow as tf
            
            # Load the model
            model = tf.keras.models.load_model(model_path)
            
            # Create converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Configure quantization
            if quantization_type == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = ModelOptimizer._representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save quantized model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model quantized and saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            return False
    
    @staticmethod
    def _representative_dataset():
        """Representative dataset for quantization"""
        for _ in range(100):
            # Generate sample data matching your input shape
            yield [np.random.random((1, 128, 128, 1)).astype(np.float32)]
    
    @staticmethod
    def benchmark_model(inference_engine: PrivacyPreservingInferenceEngine, 
                       input_shape: Tuple[int, ...], 
                       num_runs: int = 100) -> Dict:
        """
        Benchmark model performance locally
        """
        try:
            # Generate test data
            test_input = np.random.random(input_shape).astype(np.float32)
            
            # Warm up
            for _ in range(10):
                inference_engine.predict(test_input)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = inference_engine.predict(test_input)
                end_time = time.time()
                
                if result['success']:
                    times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if times:
                return {
                    'avg_inference_time_ms': np.mean(times),
                    'std_inference_time_ms': np.std(times),
                    'min_inference_time_ms': np.min(times),
                    'max_inference_time_ms': np.max(times),
                    'throughput_fps': 1000 / np.mean(times),
                    'num_successful_runs': len(times),
                    'privacy_status': 'local_only'
                }
            else:
                return {'error': 'No successful inference runs'}
                
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {'error': str(e)}

# Factory function for creating inference engines
def create_inference_engine(engine_type: str = 'tflite', **kwargs) -> PrivacyPreservingInferenceEngine:
    """
    Factory function to create inference engines
    """
    if engine_type.lower() == 'tflite':
        return TensorFlowLiteEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported engine type: {engine_type}")

# Example usage and testing
if __name__ == "__main__":
    # Example of setting up privacy-preserving inference
    engine = create_inference_engine('tflite', enable_quantization=True)
    
    # Load model (replace with actual model path)
    model_loaded = engine.load_model("models/acoustic_classifier.tflite")
    
    if model_loaded:
        # Create audio processor
        processor = PrivacyPreservingAudioProcessor(engine)
        
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