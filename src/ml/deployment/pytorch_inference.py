import torch
import numpy as np
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyTorchScriptInferenceEngine:
    """
    Privacy-preserving inference engine using TorchScript models (PyTorch)
    """
    def __init__(self):
        self.model = None
        self.metadata = None
        self.model_loaded = False
        self.model_size_bytes = 0
        self.inference_count = 0
        self.total_inference_time = 0

    def load_model(self, model_path: str) -> bool:
        try:
            # Normalize path separators
            model_path = os.path.normpath(model_path)
            
            # Handle different metadata file naming patterns
            meta_path = model_path.replace('.pt', '_metadata.json')
            if not os.path.exists(meta_path):
                # Try alternative naming pattern
                base_name = os.path.splitext(os.path.basename(model_path))[0]
                if base_name.endswith('_scripted'):
                    base_name = base_name.replace('_scripted', '')
                meta_path = os.path.join(os.path.dirname(model_path), f"{base_name}_metadata.json")
            
            meta_path = os.path.normpath(meta_path)
            
            logger.info(f"Looking for model at: {model_path}")
            logger.info(f"Looking for metadata at: {meta_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            if not os.path.exists(meta_path):
                logger.error(f"Metadata file not found: {meta_path}")
                return False
                
            self.model = torch.jit.load(model_path, map_location='cpu')
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
            self.model.eval()
            self.model_size_bytes = os.path.getsize(model_path)
            self.model_loaded = True
            logger.info(f"TorchScript model loaded: {model_path} ({self.model_size_bytes} bytes)")
            return True
        except Exception as e:
            logger.error(f"Error loading TorchScript model: {e}")
            return False

    def predict(self, input_data: np.ndarray) -> dict:
        if not self.model_loaded:
            return {'error': 'Model not loaded', 'privacy_status': 'local_only', 'success': False}
        try:
            import time
            start_time = time.time()
            x = torch.tensor(input_data, dtype=torch.float32)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
            inference_time = (time.time() - start_time) * 1000
            self.inference_count += 1
            self.total_inference_time += inference_time
            class_labels = self.metadata.get('class_labels', [])
            pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else 'unknown'
            return {
                'predictions': probs.tolist(),
                'predicted_class': pred_idx,
                'predicted_label': pred_label,
                'confidence': confidence,
                'inference_time_ms': inference_time,
                'privacy_status': 'local_only',
                'data_transmitted': False,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'class_labels': class_labels
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {'error': str(e), 'privacy_status': 'local_only', 'success': False}

    def get_model_info(self) -> dict:
        if not self.model_loaded:
            return {'error': 'Model not loaded'}
        return {
            'model_type': 'torchscript',
            'feature_size': self.metadata.get('feature_size', 0),
            'num_classes': self.metadata.get('class_labels', []),
            'accuracy': self.metadata.get('val_accuracy', 0),
            'model_size_bytes': self.model_size_bytes,
            'inference_count': self.inference_count,
            'avg_inference_time_ms': self.total_inference_time / max(self.inference_count, 1),
            'privacy_status': 'local_only',
            'class_labels': self.metadata.get('class_labels', [])
        }

    def cleanup(self):
        self.model = None
        self.metadata = None
        self.model_loaded = False
        logger.info("PyTorchScript inference engine cleaned up")

# Example usage
def example():
    engine = PyTorchScriptInferenceEngine()
    model_path = os.path.join('ml_models', 'mlp_model_scripted.pt')
    if not engine.load_model(model_path):
        print("Model not loaded!")
        return
    # Simulate audio features
    features = np.random.random(engine.metadata['feature_size']).astype(np.float32)
    result = engine.predict(features)
    print("Result:", result)
    engine.cleanup()

if __name__ == "__main__":
    example() 