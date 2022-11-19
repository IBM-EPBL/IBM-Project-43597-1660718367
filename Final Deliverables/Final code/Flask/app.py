# -*- coding: utf-8 -*-
import os
import copy
import itertools
import json
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from flask import redirect, url_for
from PIL import Image

app=Flask(__name__)

json_file = open("C:/Users/admin/Desktop/Flask/final.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("C:/Users/admin/Desktop/Flask/final.h5")

@app.route('/')
def index():
    return render_template('sample.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/reg')
def reg():
    return render_template('reg.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, '', f.filename)
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(loaded_model.predict(x), axis=-1)
        op = ['Corpse Flower','Great Indian Bustard Bird', 'Lady Slipper Orchid Flower',
              'Pangolin Mammal', 'Senenca White Deer Mammal','Spoon Billed Sandpiper Bird']
        text = op[pred[0]]
        return render_template('predict.html', value=text)
    
if __name__ == "__main__":
    app.run(debug=True)

