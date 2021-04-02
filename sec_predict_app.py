# -*- coding: utf-8 -*-
import base64
import numpy as np
import io
from PIL import Image

import tensorflow as tf
from flask import request
from flask import jsonify
from flask import Flask, render_template
from flask_cors import CORS,cross_origin
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
#original imports
"""
from tensorflow import keras

from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
"""

#print(tf.__version__)


app = Flask(__name__)
cors=CORS(app)
app.config['CORS_HEADERS']='Content-Type'



def get_model():
    global model
    model = load_model('C:/Users/Russ/Desktop/Flask Tutorial/sec_local/sec_model.h5')
    print(" * Model loaded!")
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print(" * Loading Keras model...")
get_model()   

@app.route("/program")
def program():
    return render_template('my_html_file.html')
    
@app.route("/test")
def test():
    return "Hello world!"

@app.route("/predict", methods=["POST"])
#@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
#    print(encoded)
    decoded = base64.b64decode(encoded)
    #print(decoded)
    
    bufferVar = io.BytesIO(decoded)
 #   print(bufferVar)
    
    other_image = Image.open(bufferVar)
    
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image).tolist()
    response = {
        'prediction': {
#              'dog': prediction[0][0],
#              'cat': prediction[0][1]  
                
            'alabama': prediction[0][0],
            'arkansas': prediction[0][1],
            'auburn': prediction[0][2],
            'florida': prediction[0][3],
            'georgia': prediction[0][4],
            'kentucky': prediction[0][5],
            'lsu': prediction[0][6],
            'mississippist': prediction[0][7],
            'missouri': prediction[0][8],
            'olemiss': prediction[0][9],
            'southcarolina': prediction[0][10],
            'tamu': prediction[0][11],
            'tennessee': prediction[0][12],
            'vanderbilt': prediction[0][13],
            
        }
    }
#    response = {'auburn': 0.103456346, 'florida': 0.9348723457}
    print (response)
    return jsonify(response)

if __name__ == '__main__':
      app.run()
#print('before load model')


#if __name__ == '__main__':
#    app.run()
     