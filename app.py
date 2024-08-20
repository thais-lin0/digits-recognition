from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from PIL import Image
import io
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('xgb_tuned.pkl')

def preprocess_image(image):
    """
    Preprocess the input image to match the format expected by the model.
    - Convert to grayscale
    - Resize to 8x8 pixels
    - Invert colors to match the original digits dataset
    - Flatten into a 64-element array
    """
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((8, 8))  # Resize to 8x8 pixels
    image = np.array(image)  # Convert to NumPy array
    image = 16 - (image // 16)  # Invert colors
    image = image.flatten()  # Flatten the image to a 64-element array
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    # Create a DataFrame with the correct column names
    column_names = [str(i) for i in range(64)]  # Column names as strings "0", "1", ..., "63"
    processed_image_df = pd.DataFrame([processed_image], columns=column_names)
    
    # Make prediction
    prediction = model.predict(processed_image_df)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
