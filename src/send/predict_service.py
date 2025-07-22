import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import json

# Class labels (should match your folders/model)
CLASS_NAMES = [
    'airport_image_logmel',
    'bus_image_logmel',
    'metro_image_logmel',
    'park_image_logmel',
    'shopping_mall_image_logmel'
]

def audio_to_image(audio_path, out_image_path):
    y, sr = librosa.load(audio_path, sr=32000, mono=True)
    stft = librosa.stft(y, n_fft=4096, win_length=3072, hop_length=500)
    mel = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=256)
    log_mel = librosa.power_to_db(mel)
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    librosa.display.specshow(log_mel, sr=sr, hop_length=500, cmap='viridis', x_axis=None, y_axis=None)
    plt.savefig(out_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_image_path

def predict_image(image_path, model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    input_scale, input_zero_point = input_details[0]['quantization']
    input_data = img.astype(np.float32) / 255.0
    input_data = input_data / input_scale + input_zero_point
    input_data = np.clip(input_data, 0, 255).astype(np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]['quantization']
    output_float = output_scale * (output_data.astype(np.float32) - output_zero_point)
    predicted_index = np.argmax(output_float[0])
    predicted_confidence = float(np.max(output_float[0]))
    predicted_label = CLASS_NAMES[predicted_index] if predicted_confidence >= 0.5 else 'unknown'
    return predicted_label, predicted_confidence

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: predict_service.py <audio_path> <model_path>"}))
        sys.exit(1)
    audio_path = sys.argv[1]
    model_path = sys.argv[2]
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        img_path = tmp_img.name
    audio_to_image(audio_path, img_path)
    pred_label, confidence = predict_image(img_path, model_path)
    os.remove(img_path)
    print(json.dumps({"predicted_class": pred_label, "confidence": confidence}))

if __name__ == "__main__":
    main()
