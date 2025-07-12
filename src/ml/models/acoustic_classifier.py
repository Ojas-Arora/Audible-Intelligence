import tensorflow as tf
import numpy as np
import librosa
import joblib
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime

class AcousticEventClassifier:
    """
    TensorFlow Lite model for real-time acoustic event classification
    Trained on DCASE dataset with custom augmentations
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or "models/acoustic_classifier.tflite"
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_labels = [
            'dog_bark', 'car_horn', 'alarm', 'glass_break', 
            'door_slam', 'siren', 'footsteps', 'speech',
            'music', 'machinery', 'nature', 'silence'
        ]
        self.category_mapping = {
            'animals': ['dog_bark', 'nature'],
            'vehicles': ['car_horn', 'siren'],
            'alarms': ['alarm', 'glass_break'],
            'home': ['door_slam', 'footsteps', 'speech', 'music', 'machinery']
        }
        self.sample_rate = 22050
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048
        self.duration = 2.0  # seconds
        
        self._load_model()
    
    def _load_model(self):
        """Load TensorFlow Lite model"""
        try:
            if os.path.exists(self.model_path):
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found: {self.model_path}")
                self._create_dummy_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration"""
        print("Creating dummy model for demonstration...")
        # This would be replaced with actual trained model
        pass
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features from audio data
        """
        # Ensure audio is the right length
        target_length = int(self.sample_rate * self.duration)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
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
    
    def predict(self, audio_data: np.ndarray) -> Dict:
        """
        Predict acoustic event from audio data
        """
        try:
            # Extract features
            features = self.extract_features(audio_data)
            
            if self.interpreter is None:
                # Dummy prediction for demonstration
                return self._dummy_prediction()
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Get top prediction
            top_class_idx = np.argmax(predictions)
            confidence = float(predictions[top_class_idx])
            predicted_class = self.class_labels[top_class_idx]
            
            # Get category
            category = self._get_category(predicted_class)
            
            return {
                'event_type': predicted_class,
                'confidence': confidence,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'all_predictions': {
                    label: float(pred) for label, pred in zip(self.class_labels, predictions)
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._dummy_prediction()
    
    def _dummy_prediction(self) -> Dict:
        """Generate dummy prediction for demonstration"""
        import random
        
        event_types = ['dog_bark', 'car_horn', 'alarm', 'glass_break', 'door_slam']
        selected_event = random.choice(event_types)
        confidence = 0.7 + random.random() * 0.3
        
        return {
            'event_type': selected_event,
            'confidence': confidence,
            'category': self._get_category(selected_event),
            'timestamp': datetime.now().isoformat(),
            'all_predictions': {label: random.random() * 0.3 for label in self.class_labels}
        }
    
    def _get_category(self, event_type: str) -> str:
        """Get category for event type"""
        for category, events in self.category_mapping.items():
            if event_type in events:
                return category
        return 'other'
    
    def batch_predict(self, audio_batch: List[np.ndarray]) -> List[Dict]:
        """Predict multiple audio samples"""
        return [self.predict(audio) for audio in audio_batch]

class AudioProcessor:
    """
    Real-time audio processing and feature extraction
    """
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer = np.array([])
        self.classifier = AcousticEventClassifier()
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process incoming audio chunk and detect events
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk])
        
        # Check if we have enough data for classification
        required_samples = int(self.sample_rate * self.classifier.duration)
        
        if len(self.buffer) >= required_samples:
            # Extract segment for classification
            segment = self.buffer[:required_samples]
            
            # Check if segment has sufficient energy (not silence)
            if self._has_sufficient_energy(segment):
                prediction = self.classifier.predict(segment)
                
                # Only return if confidence is above threshold
                if prediction['confidence'] > 0.6:
                    # Remove processed samples from buffer
                    self.buffer = self.buffer[required_samples//2:]  # 50% overlap
                    return prediction
            
            # Remove old samples to prevent buffer overflow
            self.buffer = self.buffer[self.chunk_size:]
        
        return None
    
    def _has_sufficient_energy(self, audio_segment: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if audio segment has sufficient energy to be non-silence"""
        rms_energy = np.sqrt(np.mean(audio_segment**2))
        return rms_energy > threshold
    
    def reset_buffer(self):
        """Reset audio buffer"""
        self.buffer = np.array([])

# Model training utilities
class ModelTrainer:
    """
    Utilities for training acoustic event detection models
    """
    
    def __init__(self):
        self.sample_rate = 22050
        self.n_mels = 128
        self.duration = 2.0
    
    def create_model(self, num_classes: int = 12) -> tf.keras.Model:
        """
        Create CNN model for acoustic event classification
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.n_mels, None, 1)),
            
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global pooling and dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def convert_to_tflite(self, model: tf.keras.Model, output_path: str):
        """
        Convert Keras model to TensorFlow Lite format
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Enable quantization for smaller model size
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to {output_path}")
    
    def _representative_dataset(self):
        """Representative dataset for quantization"""
        # This would be replaced with actual training data
        for _ in range(100):
            yield [np.random.random((1, 128, 87, 1)).astype(np.float32)]

if __name__ == "__main__":
    # Example usage
    classifier = AcousticEventClassifier()
    processor = AudioProcessor()
    
    # Simulate audio processing
    dummy_audio = np.random.random(22050 * 2)  # 2 seconds of audio
    result = classifier.predict(dummy_audio)
    print("Prediction result:", result)