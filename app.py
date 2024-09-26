from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model('model/eye_disease_model.h5')

# Ensure uploads directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Dictionary to store disease information
disease_info = {
    'Bulging eyes': {
        'description': 'Abnormal protrusion of one or both eyes, often caused by thyroid-related conditions.',
        'symptoms': ['Protruding eyes', 'Eye dryness', 'Difficulty closing eyes', 'Eye pain or pressure', 'Double vision'],
        'causes': ['Thyroid eye disease', 'Tumors', 'Inflammation'],
        'treatments': ['Treat thyroid disorder', 'Artificial tears', 'Surgery if severe']
    },
    'Cataracts': {
        'description': 'Clouding of the eye\'s natural lens, leading to vision impairment.',
        'symptoms': ['Blurry vision', 'Difficulty seeing at night', 'Faded colors', 'Sensitivity to light'],
        'causes': ['Aging', 'Eye injury or surgery', 'Genetic factors'],
        'treatments': ['Surgery to remove the clouded lens']
    },
    'Crossed eyes': {
        'description': 'Misalignment of the eyes, where the eyes do not look in the same direction at the same time.',
        'symptoms': ['Eyes not aligned', 'Double vision', 'Difficulty focusing'],
        'causes': ['Imbalance in eye muscles', 'Nerve damage', 'Congenital condition'],
        'treatments': ['Glasses or contact lenses', 'Eye muscle surgery', 'Eye exercises']
    },
    'Glaucoma': {
        'description': 'A group of eye conditions that damage the optic nerve, often due to high eye pressure.',
        'symptoms': ['Gradual loss of peripheral vision', 'Blurred vision', 'Eye pain'],
        'causes': ['Elevated eye pressure', 'Genetics', 'Age-related factors'],
        'treatments': ['Eye drops', 'Laser treatment', 'Surgery']
    },
    'Uveitis': {
        'description': 'Inflammation of the uvea, the middle layer of the eye.',
        'symptoms': ['Red eyes', 'Blurred vision', 'Sensitivity to light', 'Floaters in vision'],
        'causes': ['Autoimmune disorders', 'Infections', 'Eye injury'],
        'treatments': ['Steroid eye drops', 'Medications to treat infection or autoimmune diseases']
    }
}

def get_disease_info(predicted_disease):
    return disease_info.get(predicted_disease, {})

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Read the uploaded image and preprocess it for the model
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # Make the prediction
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            disease_classes = ['Bulging eyes', 'Cataracts', 'Crossed eyes', 'Glaucoma', 'Uveitis']
            predicted_disease = disease_classes[predicted_class]

            # Get disease info
            disease_details = get_disease_info(predicted_disease)

            # Generate image URL
            img_url = url_for('static', filename=f'uploads/{file.filename}')

            # Render the result template
            return render_template('result.html', disease=predicted_disease, details=disease_details, img_url=img_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
