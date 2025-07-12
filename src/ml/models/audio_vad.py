import numpy as np
import librosa
from typing import Tuple, List
import tensorflow as tf
from scipy import signal

class VoiceActivityDetector:
    """
    Voice Activity Detection using energy-based and spectral features
    Optimized for mobile deployment
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.frame_length = 1024
        self.hop_length = 512
        self.energy_threshold = 0.01
        self.spectral_threshold = 0.1
        self.zero_crossing_threshold = 0.1
        
    def detect_activity(self, audio_data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice/sound activity in audio segment
        Returns: (is_active, confidence_score)
        """
        if len(audio_data) == 0:
            return False, 0.0
        
        # Normalize audio
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Calculate features
        energy_score = self._calculate_energy(audio_data)
        spectral_score = self._calculate_spectral_centroid(audio_data)
        zcr_score = self._calculate_zero_crossing_rate(audio_data)
        
        # Combine scores
        combined_score = (energy_score * 0.5 + 
                         spectral_score * 0.3 + 
                         zcr_score * 0.2)
        
        is_active = combined_score > 0.3
        
        return is_active, float(combined_score)
    
    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy"""
        rms_energy = np.sqrt(np.mean(audio_data**2))
        return min(rms_energy / self.energy_threshold, 1.0)
    
    def _calculate_spectral_centroid(self, audio_data: np.ndarray) -> float:
        """Calculate spectral centroid"""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )[0]
            
            mean_centroid = np.mean(spectral_centroids)
            normalized_centroid = mean_centroid / (self.sample_rate / 2)
            
            return min(normalized_centroid / self.spectral_threshold, 1.0)
        except:
            return 0.0
    
    def _calculate_zero_crossing_rate(self, audio_data: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        try:
            zcr = librosa.feature.zero_crossing_rate(
                audio_data, 
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]
            
            mean_zcr = np.mean(zcr)
            return min(mean_zcr / self.zero_crossing_threshold, 1.0)
        except:
            return 0.0

class AudioSegmenter:
    """
    Segment audio into meaningful chunks for processing
    """
    
    def __init__(self, sample_rate: int = 22050, segment_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(sample_rate * segment_duration)
        self.vad = VoiceActivityDetector(sample_rate)
        
    def segment_audio(self, audio_data: np.ndarray, overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment audio into overlapping chunks
        """
        segments = []
        hop_samples = int(self.segment_samples * (1 - overlap))
        
        for start in range(0, len(audio_data) - self.segment_samples + 1, hop_samples):
            segment = audio_data[start:start + self.segment_samples]
            
            # Only include segments with activity
            is_active, confidence = self.vad.detect_activity(segment)
            if is_active and confidence > 0.2:
                segments.append(segment)
        
        return segments
    
    def extract_events(self, audio_data: np.ndarray, min_duration: float = 0.5) -> List[Tuple[int, int]]:
        """
        Extract event boundaries from audio
        Returns list of (start_sample, end_sample) tuples
        """
        # Frame-wise activity detection
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(
            audio_data, 
            frame_length=frame_length, 
            hop_length=hop_length,
            axis=0
        )
        
        activity_frames = []
        for frame in frames.T:
            is_active, _ = self.vad.detect_activity(frame)
            activity_frames.append(is_active)
        
        # Find continuous active regions
        events = []
        start_frame = None
        
        for i, is_active in enumerate(activity_frames):
            if is_active and start_frame is None:
                start_frame = i
            elif not is_active and start_frame is not None:
                # End of active region
                duration_frames = i - start_frame
                duration_seconds = duration_frames * hop_length / self.sample_rate
                
                if duration_seconds >= min_duration:
                    start_sample = start_frame * hop_length
                    end_sample = i * hop_length
                    events.append((start_sample, end_sample))
                
                start_frame = None
        
        # Handle case where audio ends during active region
        if start_frame is not None:
            duration_frames = len(activity_frames) - start_frame
            duration_seconds = duration_frames * hop_length / self.sample_rate
            
            if duration_seconds >= min_duration:
                start_sample = start_frame * hop_length
                end_sample = len(audio_data)
                events.append((start_sample, end_sample))
        
        return events

class AudioPreprocessor:
    """
    Audio preprocessing pipeline for ML models
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline
        """
        # Normalize
        audio_data = self._normalize(audio_data)
        
        # Remove DC offset
        audio_data = self._remove_dc_offset(audio_data)
        
        # Apply pre-emphasis filter
        audio_data = self._pre_emphasis(audio_data)
        
        # Noise reduction (simple spectral subtraction)
        audio_data = self._noise_reduction(audio_data)
        
        return audio_data
    
    def _normalize(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def _remove_dc_offset(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove DC offset"""
        return audio_data - np.mean(audio_data)
    
    def _pre_emphasis(self, audio_data: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter"""
        return np.append(audio_data[0], audio_data[1:] - alpha * audio_data[:-1])
    
    def _noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple noise reduction using spectral subtraction"""
        try:
            # Compute STFT
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            
            # Ensure non-negative values
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
        except:
            # Return original audio if processing fails
            return audio_data

if __name__ == "__main__":
    # Example usage
    vad = VoiceActivityDetector()
    segmenter = AudioSegmenter()
    preprocessor = AudioPreprocessor()
    
    # Test with dummy audio
    dummy_audio = np.random.random(44100)  # 2 seconds at 22050 Hz
    
    # Preprocess
    processed_audio = preprocessor.preprocess(dummy_audio)
    
    # Detect activity
    is_active, confidence = vad.detect_activity(processed_audio)
    print(f"Activity detected: {is_active}, Confidence: {confidence:.3f}")
    
    # Segment audio
    segments = segmenter.segment_audio(processed_audio)
    print(f"Found {len(segments)} active segments")