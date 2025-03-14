import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from cnn import Model

app = Flask(__name__)

# Load the trained model
model_path = 'saved_model.h5'  # Adjust to your actual path
model = Model.load_classifier(model_path)

# Define the classes for prediction
classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the frame from the request
        data = request.json
        frame_base64 = data.get("frame")

        if not frame_base64:
            return jsonify({"error": "No frame provided"}), 400

        # Decode the base64 frame
        frame_data = np.frombuffer(
            cv2.imdecode(np.frombuffer(base64.b64decode(frame_base64), np.uint8), cv2.IMREAD_COLOR), np.uint8)
        image = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        # Preprocess the image for the model
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))  # Model input size
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))  # Add batch and channel dimensions

        # Predict using the model
        pred = model.predict(gray)
        predicted_class = classes[np.argmax(pred)]
        confidence = np.max(pred)

        return jsonify({
            "prediction": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Ensure the Flask app runs only when this file is executed
    app.run(host='127.0.0.1', port=5000, debug=True)
