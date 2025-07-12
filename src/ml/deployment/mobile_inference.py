import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
import threading
import queue
from abc import ABC, abstractmethod

class MobileInferenceEngine(ABC):
    """
    Abstract base class for mobile inference engines
    """
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        pass
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        pass

class TensorFlowLiteEngine(MobileInferenceEngine):
    """
    TensorFlow Lite inference engine for mobile deployment
    """
    
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_loaded = False
        
    def load_model(self, model_path: str) -> bool:
        """
        Load TensorFlow Lite model
        """
        try:
            import tensorflow as tf
            
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_loaded = True
            print(f"TensorFlow Lite model loaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading TensorFlow Lite model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """
        Run inference on input data
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            start_time = time.time()
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            return {
                'predictions': output_data.tolist(),
                'inference_time_ms': inference_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        return {
            'input_shape': self.input_details[0]['shape'].tolist(),
            'input_dtype': str(self.input_details[0]['dtype']),
            'output_shape': self.output_details[0]['shape'].tolist(),
            'output_dtype': str(self.output_details[0]['dtype']),
            'model_size_bytes': self.interpreter.get_tensor_details()[0].get('quantization', {}).get('scale', 'N/A')
        }

class ONNXEngine(MobileInferenceEngine):
    """
    ONNX Runtime inference engine
    """
    
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = None
        self.model_loaded = False
        
    def load_model(self, model_path: str) -> bool:
        """
        Load ONNX model
        """
        try:
            import onnxruntime as ort
            
            # Configure for mobile/edge deployment
            providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                model_path, 
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            print(f"ONNX model loaded: {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> Dict:
        """
        Run ONNX inference
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        try:
            start_time = time.time()
            
            # Prepare input
            input_dict = {self.input_name: input_data.astype(np.float32)}
            
            # Run inference
            outputs = self.session.run(self.output_names, input_dict)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            return {
                'predictions': outputs[0].tolist(),
                'inference_time_ms': inference_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }
    
    def get_model_info(self) -> Dict:
        """
        Get ONNX model information
        """
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        return {
            'input_shape': input_info.shape,
            'input_dtype': input_info.type,
            'output_shape': output_info.shape,
            'output_dtype': output_info.type
        }

class RealTimeAudioProcessor:
    """
    Real-time audio processing with ML inference
    """
    
    def __init__(self, 
                 inference_engine: MobileInferenceEngine,
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_size: int = 10):
        
        self.inference_engine = inference_engine
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        
        # Audio buffer and processing
        self.audio_buffer = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue()
        
        # Processing thread
        self.processing_thread = None
        self.is_processing = False
        
        # Feature extraction
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        
        # Performance metrics
        self.metrics = {
            'total_chunks_processed': 0,
            'average_inference_time': 0.0,
            'dropped_chunks': 0,
            'processing_errors': 0
        }
    
    def start_processing(self):
        """
        Start real-time audio processing
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time audio processing started")
    
    def stop_processing(self):
        """
        Stop real-time audio processing
        """
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
        
        print("Real-time audio processing stopped")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Add audio chunk to processing queue
        """
        try:
            self.audio_buffer.put_nowait(audio_chunk)
            return True
        except queue.Full:
            self.metrics['dropped_chunks'] += 1
            return False
    
    def get_latest_result(self) -> Optional[Dict]:
        """
        Get latest processing result
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """
        Main processing loop
        """
        while self.is_processing:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_buffer.get(timeout=0.1)
                
                # Process chunk
                result = self._process_audio_chunk(audio_chunk)
                
                # Store result
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Remove oldest result to make space
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass
                
                # Update metrics
                self.metrics['total_chunks_processed'] += 1
                
                if 'inference_time_ms' in result:
                    # Update average inference time
                    current_avg = self.metrics['average_inference_time']
                    new_time = result['inference_time_ms']
                    count = self.metrics['total_chunks_processed']
                    
                    self.metrics['average_inference_time'] = (
                        (current_avg * (count - 1) + new_time) / count
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                self.metrics['processing_errors'] += 1
                print(f"Processing error: {e}")
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """
        Process single audio chunk
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_chunk)
            
            # Run inference
            result = self.inference_engine.predict(features)
            
            # Add timestamp
            result['timestamp'] = time.time()
            result['chunk_size'] = len(audio_chunk)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': time.time(),
                'success': False
            }
    
    def get_metrics(self) -> Dict:
        """
        Get processing metrics
        """
        return self.metrics.copy()

class AudioFeatureExtractor:
    """
    Extract features from audio for ML inference
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048
        self.duration = 2.0
        
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features
        """
        try:
            import librosa
            
            # Ensure consistent length
            target_length = int(self.sample_rate * self.duration)
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
            
            # Normalize
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
            
            # Reshape for model input
            return log_mel_spec.reshape(1, self.n_mels, -1, 1).astype(np.float32)
            
        except ImportError:
            # Fallback feature extraction
            return self._simple_feature_extraction(audio)
    
    def _simple_feature_extraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Simple feature extraction without librosa
        """
        # Basic spectral features using FFT
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Create mel-like filterbank
        n_filters = self.n_mels
        filter_size = len(magnitude) // n_filters
        
        mel_features = []
        for i in range(n_filters):
            start_idx = i * filter_size
            end_idx = (i + 1) * filter_size
            filter_energy = np.mean(magnitude[start_idx:end_idx])
            mel_features.append(filter_energy)
        
        # Convert to log scale and normalize
        mel_features = np.array(mel_features)
        log_mel_features = np.log(mel_features + 1e-8)
        normalized_features = (log_mel_features - np.mean(log_mel_features)) / (np.std(log_mel_features) + 1e-8)
        
        # Reshape to match expected input format
        # Assuming time dimension of ~87 frames for 2 seconds
        time_frames = 87
        features_2d = np.tile(normalized_features.reshape(-1, 1), (1, time_frames))
        
        return features_2d.reshape(1, self.n_mels, time_frames, 1).astype(np.float32)

class ModelOptimizer:
    """
    Optimize models for mobile deployment
    """
    
    @staticmethod
    def quantize_tflite_model(model_path: str, output_path: str, quantization_type: str = 'dynamic') -> bool:
        """
        Quantize TensorFlow Lite model
        """
        try:
            import tensorflow as tf
            
            # Load model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            
            # Convert back to Keras model (if possible) for re-quantization
            # This is a simplified approach; in practice, you'd start from the original Keras model
            
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path.replace('.tflite', ''))
            
            if quantization_type == 'dynamic':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization_type == 'int8':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            # Convert
            quantized_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
            
            print(f"Quantized model saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error quantizing model: {e}")
            return False
    
    @staticmethod
    def benchmark_model(inference_engine: MobileInferenceEngine, 
                       input_shape: Tuple[int, ...], 
                       num_runs: int = 100) -> Dict:
        """
        Benchmark model performance
        """
        inference_times = []
        memory_usage = []
        
        # Generate dummy input
        dummy_input = np.random.random(input_shape).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            inference_engine.predict(dummy_input)
        
        # Benchmark runs
        for i in range(num_runs):
            start_time = time.time()
            result = inference_engine.predict(dummy_input)
            end_time = time.time()
            
            if result.get('success', False):
                inference_times.append((end_time - start_time) * 1000)  # ms
        
        if not inference_times:
            return {'error': 'No successful inferences'}
        
        return {
            'average_inference_time_ms': np.mean(inference_times),
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'std_inference_time_ms': np.std(inference_times),
            'successful_runs': len(inference_times),
            'total_runs': num_runs,
            'success_rate': len(inference_times) / num_runs
        }

if __name__ == "__main__":
    # Example usage
    print("Mobile Inference Engine Test")
    
    # Test TensorFlow Lite engine
    tflite_engine = TensorFlowLiteEngine()
    
    # Test real-time processor
    processor = RealTimeAudioProcessor(tflite_engine)
    
    # Generate dummy audio data
    sample_rate = 22050
    duration = 2.0
    dummy_audio = np.random.random(int(sample_rate * duration)).astype(np.float32)
    
    # Test feature extraction
    feature_extractor = AudioFeatureExtractor(sample_rate)
    features = feature_extractor.extract_features(dummy_audio)
    
    print(f"Extracted features shape: {features.shape}")
    
    # Test processing
    processor.start_processing()
    
    # Add some audio chunks
    chunk_size = 1024
    for i in range(0, len(dummy_audio), chunk_size):
        chunk = dummy_audio[i:i+chunk_size]
        processor.add_audio_chunk(chunk)
        time.sleep(0.01)  # Simulate real-time
    
    # Wait a bit for processing
    time.sleep(1.0)
    
    # Get results
    while True:
        result = processor.get_latest_result()
        if result is None:
            break
        print(f"Processing result: {result}")
    
    # Get metrics
    metrics = processor.get_metrics()
    print(f"Processing metrics: {metrics}")
    
    processor.stop_processing()
    
    print("Mobile inference test completed")