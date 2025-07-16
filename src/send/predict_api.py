import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_service import audio_to_image, predict_image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'bc_resnet_40_int8.tflite')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
        audio_file.save(audio_path)

    # Generate spectrogram image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        img_path = tmp_img.name
    try:
        audio_to_image(audio_path, img_path)
        pred_label, confidence = predict_image(img_path, MODEL_PATH)
    except Exception as e:
        os.remove(audio_path)
        os.remove(img_path)
        return jsonify({'error': str(e)}), 500
    os.remove(audio_path)
    os.remove(img_path)
    return jsonify({'predicted_class': pred_label, 'confidence': confidence})

@app.route('/predict-from-path', methods=['POST'])
def predict_from_path():
    # Use the same audio path as in image_generate.ipynb
    # Read the input audio path from a file (set by image_generate or user)
    input_audio_path_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'input_audio_path.txt'))
    if os.path.exists(input_audio_path_file):
        with open(input_audio_path_file, 'r') as f:
            audio_path = f.read().strip()
    else:
        # fallback: use previous default
        audio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_audio', 'airport', 'airport-barcelona-203-6122-0-b.wav'))
    print(f"Predicting for audio file: {audio_path}")  # DEBUG
    model_path = MODEL_PATH
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
        img_path = tmp_img.name
    try:
        audio_to_image(audio_path, img_path)
        pred_label, confidence = predict_image(img_path, model_path)
    except Exception as e:
        print(f"Error in /predict-from-path: {e}")
        os.remove(img_path)
        return jsonify({'error': str(e)}), 500
    os.remove(img_path)
    return jsonify({'predicted_class': pred_label, 'confidence': confidence})

if __name__ == '__main__':
    print('==============================')
    print('Flask backend is running and ready to accept connections!')
    print('Visit: http://127.0.0.1:5001 or http://localhost:5001')
    print('==============================')
    app.run(host='0.0.0.0', port=5001, debug=True)
