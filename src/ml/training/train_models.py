import tensorflow as tf
import numpy as np
import librosa
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import pandas as pd

class AcousticDatasetLoader:
    """
    Load and preprocess acoustic event datasets
    Supports DCASE, ESC-50, and custom datasets
    """
    
    def __init__(self, data_dir: str, sample_rate: int = 22050):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = 2.0  # seconds
        self.n_mels = 128
        self.hop_length = 512
        self.n_fft = 2048
        
        # Event categories mapping
        self.category_mapping = {
            'animals': ['dog_bark', 'cat_meow', 'bird_singing', 'cow_moo'],
            'vehicles': ['car_horn', 'motorcycle', 'truck', 'airplane'],
            'alarms': ['smoke_alarm', 'car_alarm', 'burglar_alarm', 'fire_alarm'],
            'home': ['door_slam', 'footsteps', 'glass_break', 'vacuum_cleaner'],
            'nature': ['rain', 'wind', 'thunder', 'ocean_waves'],
            'urban': ['construction', 'traffic', 'crowd', 'street_music']
        }
        
    def load_dcase_dataset(self, metadata_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load DCASE dataset
        """
        # Read metadata
        metadata = pd.read_csv(metadata_file, sep='\t')
        
        features = []
        labels = []
        label_names = []
        
        for idx, row in metadata.iterrows():
            audio_file = os.path.join(self.data_dir, row['filename'])
            
            if os.path.exists(audio_file):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                    
                    # Extract features
                    mel_features = self.extract_mel_features(audio)
                    
                    features.append(mel_features)
                    labels.append(row['scene_label'])
                    
                    if row['scene_label'] not in label_names:
                        label_names.append(row['scene_label'])
                        
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        return np.array(features), np.array(labels), label_names
    
    def load_esc50_dataset(self, metadata_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load ESC-50 dataset
        """
        metadata = pd.read_csv(metadata_file)
        
        features = []
        labels = []
        label_names = list(metadata['category'].unique())
        
        for idx, row in metadata.iterrows():
            audio_file = os.path.join(self.data_dir, row['filename'])
            
            if os.path.exists(audio_file):
                try:
                    # Load audio
                    audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                    
                    # Ensure consistent length
                    target_length = int(self.sample_rate * self.duration)
                    if len(audio) > target_length:
                        audio = audio[:target_length]
                    elif len(audio) < target_length:
                        audio = np.pad(audio, (0, target_length - len(audio)))
                    
                    # Extract features
                    mel_features = self.extract_mel_features(audio)
                    
                    features.append(mel_features)
                    labels.append(row['category'])
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        return np.array(features), np.array(labels), label_names
    
    def extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram features
        """
        # Compute mel-spectrogram
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
        
        return log_mel_spec
    
    def augment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation techniques
        """
        augmented_samples = [audio]  # Original
        
        # Time stretching
        try:
            stretched = librosa.effects.time_stretch(audio, rate=0.9)
            if len(stretched) == len(audio):
                augmented_samples.append(stretched)
        except:
            pass
        
        # Pitch shifting
        try:
            pitched = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
            augmented_samples.append(pitched)
        except:
            pass
        
        # Add noise
        noise_factor = 0.005
        noisy = audio + noise_factor * np.random.randn(len(audio))
        augmented_samples.append(noisy)
        
        # Volume scaling
        scaled = audio * np.random.uniform(0.7, 1.3)
        augmented_samples.append(scaled)
        
        return augmented_samples

class AcousticModelTrainer:
    """
    Train acoustic event detection models
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 87, 1)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def create_cnn_model(self, num_classes: int) -> tf.keras.Model:
        """
        Create CNN model for acoustic classification
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global pooling and classification
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_attention_model(self, num_classes: int) -> tf.keras.Model:
        """
        Create attention-based model for acoustic classification
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Convolutional feature extraction
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        
        # Reshape for attention
        shape = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Reshape((shape[1] * shape[2], shape[3]))(x)
        
        # Self-attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=64
        )(x, x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(attention)
        
        # Classification head
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   num_classes: int,
                   epochs: int = 100,
                   batch_size: int = 32,
                   model_type: str = 'cnn') -> tf.keras.Model:
        """
        Train the acoustic classification model
        """
        # Create model
        if model_type == 'attention':
            self.model = self.create_attention_model(num_classes)
        else:
            self.model = self.create_cnn_model(num_classes)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_acoustic_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]):
        """
        Evaluate trained model
        """
        if self.model is None:
            print("No model to evaluate. Train a model first.")
            return
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Training history
        if self.history:
            self.plot_training_history()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def convert_to_tflite(self, output_path: str = 'acoustic_model.tflite'):
        """
        Convert model to TensorFlow Lite format
        """
        if self.model is None:
            print("No model to convert. Train a model first.")
            return
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TensorFlow Lite model saved to {output_path}")
        
        # Model info
        model_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"Model size: {model_size:.2f} MB")

def main():
    """
    Main training pipeline
    """
    # Configuration
    DATA_DIR = "data/acoustic_events"
    METADATA_FILE = "data/metadata.csv"
    
    # Initialize components
    loader = AcousticDatasetLoader(DATA_DIR)
    trainer = AcousticModelTrainer()
    
    print("Loading dataset...")
    
    # Create dummy dataset for demonstration
    # In practice, replace this with actual dataset loading
    n_samples = 1000
    n_classes = 8
    
    # Generate dummy features (mel-spectrograms)
    X = np.random.random((n_samples, 128, 87, 1)).astype(np.float32)
    
    # Generate dummy labels
    y_labels = np.random.randint(0, n_classes, n_samples)
    y = tf.keras.utils.to_categorical(y_labels, n_classes)
    
    class_names = ['dog_bark', 'car_horn', 'alarm', 'glass_break', 
                   'door_slam', 'siren', 'footsteps', 'speech']
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\nTraining model...")
    model = trainer.train_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=n_classes,
        epochs=50,
        batch_size=32,
        model_type='cnn'
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    trainer.evaluate_model(X_test, y_test, class_names)
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    trainer.convert_to_tflite('models/acoustic_classifier.tflite')
    
    # Save model info
    model_info = {
        'classes': class_names,
        'input_shape': [128, 87, 1],
        'sample_rate': 22050,
        'duration': 2.0,
        'n_mels': 128,
        'hop_length': 512,
        'n_fft': 2048
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()