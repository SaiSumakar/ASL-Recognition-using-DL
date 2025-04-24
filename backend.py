# import os
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from cnn import Model

# app = Flask(__name__)

# # Load the trained model
# model_path = 'saved_model.h5'  # Adjust to your actual path
# model = Model.load_classifier(model_path)

# # Define the classes for prediction
# classes = [
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#     'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
# ]


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the frame from the request
#         data = request.json
#         frame_base64 = data.get("frame")

#         if not frame_base64:
#             return jsonify({"error": "No frame provided"}), 400

#         # Decode the base64 frame
#         frame_data = np.frombuffer(
#             cv2.imdecode(np.frombuffer(base64.b64decode(frame_base64), np.uint8), cv2.IMREAD_COLOR), np.uint8)
#         image = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

#         # Preprocess the image for the model
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray = cv2.resize(gray, (64, 64))  # Model input size
#         gray = gray.astype("float32") / 255.0
#         gray = np.expand_dims(gray, axis=(0, -1))  # Add batch and channel dimensions

#         # Predict using the model
#         pred = model.predict(gray)
#         predicted_class = classes[np.argmax(pred)]
#         confidence = np.max(pred)

#         return jsonify({
#             "prediction": predicted_class,
#             "confidence": float(confidence)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     # Ensure the Flask app runs only when this file is executed
#     app.run(host='127.0.0.1', port=5000, debug=True)



# from flask import Flask, request, jsonify
# # from your_ml_library import load_model, preprocess_image, predict_sign

# from tensorflow.keras.models import load_model

# # Initialize Flask app
# app = Flask(__name__)

# # Load your model
# model = load_model('saved_model.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         if 'frame' not in data:
#             return jsonify({'error': 'No frame provided'}), 400
        
#         # Decode the base64 frame
#         frame_data = data['frame']
#         image = preprocess_image(frame_data)  # Custom function to decode and preprocess the image
        
#         # Predict the sign language
#         prediction = predict_sign(model, image)  # Custom function to predict using the model
        
#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify
import cv2 
import mediapipe as mp
import numpy as np
import base64
import io
from PIL import Image
from spellchecker import SpellChecker
from tensorflow.keras.models import load_model

from flask_cors import CORS

app = Flask(__name__)
CORS(app)



class Model:

  classifier = None
  def __init__(self, Type):
    self.classifier = Type
    
  def build_model(classifier):
    

    classifier.add(Convolution2D(128, (3, 3), input_shape=(64, 64, 1), activation='relu'))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Convolution2D(256, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Convolution2D(512, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.5))

    classifier.add(Flatten())

    classifier.add(Dropout(0.5))
    
    classifier.add(Dense(1024, activation='relu'))
    

    classifier.add(Dense(29, activation='softmax'))

    return classifier

  def save_classifier(path, classifier):
    classifier.save(path)

  def load_classifier(path):
    classifier = load_model(path)
    return classifier

  def predict(classes, classifier, img):
    img = cv2.resize(img, (64, 64))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0

    pred = classifier.predict(img)
    return classes[np.argmax(pred)], pred
    

class DataGatherer:

  def __init__(self, *args):
    if len(args) > 0:
      self.dir = args[0]
    elif len(args) == 0:
      self.dir = ""

  #this function loads the images along with their labels and apply pre-processing function on the images 
  # and finaly split them into train and test dataset
  def load_images(self):
    images = []
    labels = []
    index = -1
    folders = sorted(os.listdir(self.dir))
    
    for folder in folders:
      index += 1
      
      print("Loading images from folder ", folder ," has started.")
      for image in os.listdir(self.dir + '/' + folder):

        img = cv2.imread(self.dir + '/' + folder + '/' + image, 0)
        
        img = self.edge_detection(img)
        img = cv2.resize(img, (64, 64))
        img = img_to_array(img)

        images.append(img)
        labels.append(index)

    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = to_categorical(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    return x_train, x_test, y_train, y_test

  def edge_detection(self, image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


def Auto_Correct(word):
    mySpellChecker = SpellChecker()
    return mySpellChecker.correction(word)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Pre-trained Classifier
classifier = Model.load_classifier('saved_model.h5')

# Classes for Prediction
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def decode_base64_image(image_data):
    """
    Decodes a base64 string to an OpenCV image.
    """
    decoded_data = base64.b64decode(image_data)
    np_image = np.frombuffer(decoded_data, dtype=np.uint8)
    return cv2.imdecode(np_image, cv2.IMREAD_COLOR)


def preprocess_image(image):
    """
    Preprocess image for hand detection and gesture classification.
    """
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe to detect hand landmarks
    with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = [landmark.x for landmark in hand_landmarks.landmark]
                y = [landmark.y for landmark in hand_landmarks.landmark]
                center = np.array([np.mean(x) * image.shape[1], np.mean(y) * image.shape[0]]).astype('int32')

                # Crop the hand region
                cropped_image = image[max(0, center[1]-130):min(image.shape[0], center[1]+130),
                                      max(0, center[0]-130):min(image.shape[1], center[0]+130)]

                if cropped_image.size > 0:
                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    return DataGatherer().edge_detection(gray)
    return None
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_sign():
    """Handles POST requests for hand gesture prediction."""
    data = request.json
    if 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    # Decode the base64 image
    image = decode_base64_image(data['frame'])
    preprocessed_image = preprocess_image(image)

    if preprocessed_image is not None:
        # Predict the gesture
        gesture, pred = Model.predict(classes, classifier, preprocessed_image)
        if gesture in ['space', 'del']:
            return jsonify({'prediction': gesture})
        else:
            corrected_word = Auto_Correct(gesture.lower())
            return jsonify({'prediction': corrected_word})
    else:
        return jsonify({'error': 'No hand detected'}), 400


if __name__ == '__main__':
    app.run(debug=True)
