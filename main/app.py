from predict import predict_disease
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

model = tf.keras.models.load_model('main/model/crop_disease_model.h5')
label_map = np.load('main/model/label_map.npy', allow_pickle=True).item()
label_map = {v: k for k, v in label_map.items()}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = f"temp/{image_file.filename}"
    image_file.save(image_path)

    prediction = predict_disease(image_path, model, label_map)
    prediction = {key: float(value) if isinstance(value, (np.float32, np.float64)) else value
                        for key, value in prediction.items()}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)


