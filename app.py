from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model(r"C:\Users\aditi\Downloads\HandwrittenCharacterRecognition-main\HandwrittenCharacterRecognition-main\Handwritten_Character_Recognition_model.h5")


# Home route to render HTML template
@app.route('/')
def index():
  return render_template('index.html')


# Route to predict handwritten character
@app.route('/predict', methods=['POST'])
def predict():
  try:
    # Receive base64 image data from the request
    data = request.json['imageData']

    # Remove header from base64 data and decode
    base64_string = data.split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(base64_string)))

    # Convert image to grayscale and resize to 28x28
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension
    image = image.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    prediction = np.delete(prediction, predicted_class)
    predicted_class_2 = np.argmax(prediction)

    mapp = {0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57, 10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73, 19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82, 28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90, 36: 97, 37: 98, 38: 100, 39: 101, 40: 102, 41: 103, 42: 104, 43: 110, 44: 113, 45: 114, 46: 116}

    print(chr(mapp[predicted_class]), chr(mapp[predicted_class_2]))
    # Return prediction result
    return jsonify({'prediction': chr(mapp[predicted_class]), 'possibility': chr(mapp[predicted_class_2])})

  except Exception as e:
    return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
  app.run(debug=True)
