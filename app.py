from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import shutil
import wikipedia

# Keras
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/MobileNetV2.h5'

from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
model = MobileNetV2(input_shape=(224, 224,3), alpha=1.0, include_top=True, weights=None, input_tensor=None, pooling=None, classes=1000)
model.load_weights(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

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

        # Make prediction
        preds = model_predict(file_path, model)
        pred_class = decode_predictions(preds, top=5)   # ImageNet Decode
        results = []
        for i in [0, 1, 2, 3, 4]:
            results.append(str(pred_class[0][i][1]))
            results.append(' : ')
            results.append(str(pred_class[0][i][2]))
            results.append(' , ')
            
            if i == 0:
                wiki_content = wikipedia.page(wikipedia.search(str(pred_class[0][i][1])[0])).content
                results.append('{'+'Description of : '+ pred_class[0][i][1] + ',' + wiki_content[0:wiki_content.find(".")]+'}')
        
        os.remove(file_path)
        
        return " " .join(results)
    return None
    

if __name__ == '__main__':
    app.run(debug=True)

