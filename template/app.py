from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__,template_folder='template')
################################################################
################################################################
MODEL_PATH = 'Dogs-vs-Cats_model.h5'
model = load_model(MODEL_PATH)
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("\n\n\n")
        print(type(f))
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        # Process your result for human
        #pred_class = decode_predictions(preds, top=1)             # Simple argmax
        
        if preds[0][0]>0.5:
            result="Dog"
        else:
            result="Cat"
                           # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
