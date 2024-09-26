from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model/eye_disease_classifier.h5')

# Ensure uploads directory exists
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', result='No file part')

        img_file = request.files['image']
        if img_file.filename == '':
            return render_template('index.html', result='No selected file')

        if img_file and allowed_file(img_file.filename):
            img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
            img_file.save(img_path)

            # Load and preprocess the image
            img = image.load_img(img_path, target_size=(256, 256))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # Make prediction
            predictions = model.predict(img)
            class_idx = np.argmax(predictions)
            class_labels = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']  # Update as per your classes
            result = class_labels[class_idx]

    return render_template('index.html', result=result)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    app.run(debug=True)
