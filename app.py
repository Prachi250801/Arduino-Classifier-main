from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from six import BytesIO

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils

# Define a flask app
app = Flask(__name__,template_folder='template')

# Model saved with Keras model.save()
MODEL_PATH = 'saved_model.pb'
category_index = {
        1: {'id': 1, 'name': 'Arduino Uno'},
        2: {'id': 2, 'name': 'Arduino Mega'},
        3: {'id': 3, 'name': 'nodemcu'}
        }
# Load your trained model
#model = load_model(MODEL_PATH)
detect_fn = tf.saved_model.load('/')
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path (this can be local or on colossus)

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Load the COCO Label Map
    

def model_predict(image_path):
    
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    plt.rcParams['figure.figsize'] = [42, 21]
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.int32),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            agnostic_mode=False)
    return image_np_with_detections


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
        preds = model_predict(file_path)
        #print(preds)
        # Process your result for human
        #pred_class = decode_predictions(preds, top=1)             # Simple argmax
        
        im = Image.fromarray(preds)
        im.save("outputimg.jpeg")
                           # Convert to string
        return "Done"
    return None


if __name__ == '__main__':
    app.run(debug=True)
