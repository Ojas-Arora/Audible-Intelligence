import numpy as np
import librosa
from typing import Tuple, List, Optional, Union
import scipy.signal
import scipy.io.wavfile
import io
import base64

class AudioUtils:
    """
    Utility functions for audio processing and manipulation
    """
    
    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 22050, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file with specified sample rate and duration
        """
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return np.array([]), sample_rate
    
    @staticmethod
    def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 22050):
        """
        Save audio array to file
        """
        try:
            # Normalize audio to prevent clipping
            audio_normalized = audio / np.max(np.abs(audio))
            
            # Convert to 16-bit PCM
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            scipy.io.wavfile.write(file_path, sample_rate, audio_int16)
            print(f"Audio saved to {file_path}")
        except Exception as e:
            print(f"Error saving audio file {file_path}: {e}")
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        """
        if orig_sr == target_sr:
            return audio
        
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
        """
        Normalize audio using different methods
        """
        if len(audio) == 0:
            return audio
        
        if method == 'peak':
            # Peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio
        
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                return audio / rms
            return audio
        
        elif method == 'lufs':
            # Simplified LUFS-like normalization
            # This is a simplified version; proper LUFS requires more complex filtering
            target_lufs = -23.0  # EBU R128 standard
            current_lufs = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-8)
            gain_db = target_lufs - current_lufs
            gain_linear = 10**(gain_db / 20)
            return audio * gain_linear
        
        else:
            return audio
    
    @staticmethod
    def apply_window(audio: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """
        Apply windowing function to audio
        """
        if window_type == 'hann':
            window = np.hanning(len(audio))
        elif window_type == 'hamming':
            window = np.hamming(len(audio))
        elif window_type == 'blackman':
            window = np.blackman(len(audio))
        else:
            window = np.ones(len(audio))
        
        return audio * window
    
    @staticmethod
    def detect_silence(audio: np.ndarray, threshold: float = 0.01, min_duration: float = 0.5, sample_rate: int = 22050) -> List[Tuple[int, int]]:
        """
        Detect silent regions in audio
        Returns list of (start_sample, end_sample) tuples
        """
        # Calculate frame-wise energy
        frame_length = 1024
        hop_length = 512
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, axis=0)
        energy = np.array([np.sqrt(np.mean(frame**2)) for frame in frames.T])
        
        # Find silent frames
        silent_frames = energy < threshold
        
        # Find continuous silent regions
        silent_regions = []
        start_frame = None
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and start_frame is None:
                start_frame = i
            elif not is_silent and start_frame is not None:
                # End of silent region
                duration_frames = i - start_frame
                duration_seconds = duration_frames * hop_length / sample_rate
                
                if duration_seconds >= min_duration:
                    start_sample = start_frame * hop_length
                    end_sample = i * hop_length
                    silent_regions.append((start_sample, end_sample))
                
                start_frame = None
        
        return silent_regions
    
    @staticmethod
    def remove_silence(audio: np.ndarray, threshold: float = 0.01, sample_rate: int = 22050) -> np.ndarray:
        """
        Remove silent regions from audio
        """
        silent_regions = AudioUtils.detect_silence(audio, threshold, sample_rate=sample_rate)
        
        if not silent_regions:
            return audio
        
        # Create mask for non-silent regions
        mask = np.ones(len(audio), dtype=bool)
        for start, end in silent_regions:
            mask[start:end] = False
        
        return audio[mask]
    
    @staticmethod
    def split_audio(audio: np.ndarray, chunk_duration: float, sample_rate: int = 22050, overlap: float = 0.0) -> List[np.ndarray]:
        """
        Split audio into chunks with optional overlap
        """
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        chunks = []
        for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
            chunk = audio[start:start + chunk_samples]
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def calculate_spectral_features(audio: np.ndarray, sample_rate: int = 22050) -> dict:
        """
        Calculate various spectral features
        """
        features = {}
        
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
        except Exception as e:
            print(f"Error calculating spectral features: {e}")
        
        return features
    
    @staticmethod
    def apply_noise_reduction(audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply simple noise reduction using spectral subtraction
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames
            noise_frames = min(10, magnitude.shape[1] // 4)
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Spectral subtraction
            enhanced_magnitude = magnitude - noise_factor * noise_spectrum
            
            # Ensure non-negative values
            enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft)
            
            return enhanced_audio
            
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return audio
    
    @staticmethod
    def audio_to_base64(audio: np.ndarray, sample_rate: int = 22050) -> str:
        """
        Convert audio array to base64 string for web transmission
        """
        try:
            # Normalize and convert to 16-bit PCM
            audio_normalized = audio / np.max(np.abs(audio))
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
            
            # Create WAV file in memory
            buffer = io.BytesIO()
            scipy.io.wavfile.write(buffer, sample_rate, audio_int16)
            
            # Convert to base64
            audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return audio_base64
            
        except Exception as e:
            print(f"Error converting audio to base64: {e}")
            return ""
    
    @staticmethod
    def base64_to_audio(audio_base64: str) -> Tuple[np.ndarray, int]:
        """
        Convert base64 string back to audio array
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Read WAV from bytes
            buffer = io.BytesIO(audio_bytes)
            sample_rate, audio_int16 = scipy.io.wavfile.read(buffer)
            
            # Convert to float
            audio = audio_int16.astype(np.float32) / 32767.0
            
            return audio, sample_rate
            
        except Exception as e:
            print(f"Error converting base64 to audio: {e}")
            return np.array([]), 22050

class AudioAugmentation:
    """
    Audio data augmentation techniques
    """
    
    @staticmethod
    def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add random noise to audio
        """
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """
        Time stretch audio without changing pitch
        """
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except:
            return audio
    
    @staticmethod
    def pitch_shift(audio: np.ndarray, sample_rate: int, n_steps: float = 0) -> np.ndarray:
        """
        Shift pitch without changing tempo
        """
        try:
            return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        except:
            return audio
    
    @staticmethod
    def volume_change(audio: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Change volume by multiplication factor
        """
        return audio * factor
    
    @staticmethod
    def add_reverb(audio: np.ndarray, room_size: float = 0.5, damping: float = 0.5) -> np.ndarray:
        """
        Add simple reverb effect
        """
        # Simple reverb using delay and feedback
        delay_samples = int(0.05 * 22050)  # 50ms delay
        reverb_audio = np.copy(audio)
        
        for i in range(delay_samples, len(audio)):
            reverb_audio[i] += room_size * audio[i - delay_samples] * damping
        
        return reverb_audio
    
    @staticmethod
    def apply_eq(audio: np.ndarray, sample_rate: int, low_gain: float = 1.0, mid_gain: float = 1.0, high_gain: float = 1.0) -> np.ndarray:
        """
        Apply simple 3-band EQ
        """
        try:
            # Define frequency bands
            low_freq = 300
            high_freq = 3000
            
            # Design filters
            nyquist = sample_rate / 2
            
            # Low-pass for low frequencies
            low_b, low_a = scipy.signal.butter(2, low_freq / nyquist, btype='low')
            low_band = scipy.signal.filtfilt(low_b, low_a, audio)
            
            # Band-pass for mid frequencies
            mid_b, mid_a = scipy.signal.butter(2, [low_freq / nyquist, high_freq / nyquist], btype='band')
            mid_band = scipy.signal.filtfilt(mid_b, mid_a, audio)
            
            # High-pass for high frequencies
            high_b, high_a = scipy.signal.butter(2, high_freq / nyquist, btype='high')
            high_band = scipy.signal.filtfilt(high_b, high_a, audio)
            
            # Combine bands with gains
            eq_audio = (low_gain * low_band + 
                       mid_gain * mid_band + 
                       high_gain * high_band)
            
            return eq_audio
            
        except Exception as e:
            print(f"Error applying EQ: {e}")
            return audio

if __name__ == "__main__":
    # Example usage
    print("Audio Utils Test")
    
    # Generate test audio
    sample_rate = 22050
    duration = 2.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Test various functions
    print(f"Original audio shape: {test_audio.shape}")
    
    # Normalize
    normalized = AudioUtils.normalize_audio(test_audio)
    print(f"Normalized audio range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
    
    # Calculate features
    features = AudioUtils.calculate_spectral_features(test_audio, sample_rate)
    print(f"Calculated {len(features)} spectral features")
    
    # Split into chunks
    chunks = AudioUtils.split_audio(test_audio, chunk_duration=0.5, sample_rate=sample_rate)
    print(f"Split into {len(chunks)} chunks")
    
    # Apply augmentations
    augmentation = AudioAugmentation()
    
    noisy = augmentation.add_noise(test_audio, noise_factor=0.01)
    stretched = augmentation.time_stretch(test_audio, rate=1.2)
    pitched = augmentation.pitch_shift(test_audio, sample_rate, n_steps=2)
    
    print("Audio augmentation completed")
    
    # Convert to base64 and back
    audio_b64 = AudioUtils.audio_to_base64(test_audio, sample_rate)
    recovered_audio, recovered_sr = AudioUtils.base64_to_audio(audio_b64)
    
    print(f"Base64 conversion successful: {len(audio_b64)} characters")
    print(f"Recovered audio shape: {recovered_audio.shape}, SR: {recovered_sr}")