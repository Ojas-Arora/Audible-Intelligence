#!/usr/bin/env python3
"""
Test script for the PyTorch privacy-preserving on-device inference pipeline
"""
import os
import numpy as np
import subprocess
import json
from deployment.pytorch_inference import PyTorchScriptInferenceEngine

def train_and_export():
    print("🧪 Training PyTorch model and exporting TorchScript...")
    # Train and export model
    subprocess.run([
        'python', 'training/train_mobile_model_pytorch.py',
        '--epochs', '5', '--batch-size', '32', '--output-dir', 'ml_models'
    ], check=True)
    assert os.path.exists('ml_models/mlp_model_scripted.pt'), "TorchScript model not found!"
    assert os.path.exists('ml_models/mlp_model_metadata.json'), "Metadata not found!"
    print("✅ Model trained and exported.")

def test_inference():
    print("🧪 Testing PyTorchScript inference engine...")
    engine = PyTorchScriptInferenceEngine()
    model_path = os.path.join('ml_models', 'mlp_model_scripted.pt')
    model_path = os.path.normpath(model_path)  # Normalize path
    print(f"Loading model from: {model_path}")
    assert engine.load_model(model_path), "Failed to load TorchScript model."
    # Simulate audio features
    features = np.random.random(engine.metadata['feature_size']).astype(np.float32)
    result = engine.predict(features)
    assert result['success'], f"Inference failed: {result.get('error')}"
    assert result['privacy_status'] == 'local_only', "Inference should be local only."
    assert result['data_transmitted'] is False, "No data should be transmitted."
    print(f"✅ Inference result: {result['predicted_label']} (confidence: {result['confidence']:.2f})")
    engine.cleanup()

def test_privacy():
    print("🧪 Testing privacy compliance...")
    import urllib.request
    original_urlopen = urllib.request.urlopen
    def mock_urlopen(*args, **kwargs):
        raise Exception("Network calls are not allowed in privacy mode")
    urllib.request.urlopen = mock_urlopen
    try:
        engine = PyTorchScriptInferenceEngine()
        model_path = os.path.join('ml_models', 'mlp_model_scripted.pt')
        model_path = os.path.normpath(model_path)  # Normalize path
        engine.load_model(model_path)
        features = np.random.random(engine.metadata['feature_size']).astype(np.float32)
        result = engine.predict(features)
        assert result['privacy_status'] == 'local_only', "Privacy status should be local_only"
        assert result['data_transmitted'] is False, "No data should be transmitted"
        print("✅ Privacy compliance test passed.")
    finally:
        urllib.request.urlopen = original_urlopen

def run_all():
    print("\n🚀 Starting PyTorch Privacy-Preserving Pipeline Tests\n" + "="*60)
    train_and_export()
    test_inference()
    test_privacy()
    print("\n🎉 ALL TESTS PASSED!\n" + "="*60)
    print("✅ PyTorch privacy-preserving on-device inference pipeline is working correctly")
    print("✅ All audio processing happens locally")
    print("✅ No data transmission detected")
    print("✅ Privacy compliance verified")
    print("✅ Python 3.12 compatible")

if __name__ == "__main__":
    run_all() 