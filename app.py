import os
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Clean, Compatible .h5 Model ---
# This model was saved with save_format='h5' for maximum compatibility.
MODEL_PATH = 'waste_classifier_final_5.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Image classification model loaded successfully!")
except Exception as e:
    print(f"Error loading image model: {e}")
    exit()

# Define the class names in the correct order for the model's output
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image_path):
    """
    Loads an image from a file path and preprocesses it for the model.
    This function ensures the input image matches the format used during training.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of one
    # Apply the MobileNetV2-specific preprocessing
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload, prediction, and renders the result."""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        # Preprocess the image and get a prediction
        preprocessed_image = preprocess_image(filepath)
        prediction = model.predict(preprocessed_image)
        
        # Decode the prediction
        predicted_class_index = tf.argmax(prediction[0]).numpy()
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = tf.reduce_max(prediction[0]).numpy() * 100

        # Pass the results to the HTML template
        return render_template('index.html',
                               prediction=f'Prediction: {predicted_class}',
                               confidence=f'Confidence: {confidence:.2f}%',
                               uploaded_image=filepath)
    return redirect(request.url)

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
