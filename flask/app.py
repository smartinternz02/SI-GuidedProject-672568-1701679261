import re
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from flask import Flask, request, render_template
# from tensorflow.keras.models import models
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename 

#loading the model

model = load_model(r"crime.h5",compile=False)
app = Flask(__name__)


#home page
@app.route('/')
def home():
    return render_template('index.html')

#prediction page
@app.route('/prediction')
def prediction():
    return render_template('/predict.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    result = ''
    
    if request.method == 'POST':
        # try:
             
        #get the file from post request
           
        
        f = request.files['image']
    
        
        #save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("current path",basepath)
        print(f.filename)
        filepath = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print("upload folder is",filepath)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(224, 224))
        
        x = image.img_to_array(img) # converting image into array
        x = np.expand_dims(x,axis=0) # expanding  Dimensions
        pred = np.argmax(model.predict(x)) #predicting the higher probability index
        # op = ['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Burglary','Explosion','Robbery','Abuse','Shooting','Shoplifting','RoadAccident']
        op=  ['RoadAccidents', 'Assault', 'Vandalism', 'Arrest', 'Shooting', 'NormalVideos', 'Arson', 'Explosion', 'Shoplifting', 'Robbery', 'Stealing', 'Burglary', 'Abuse', 'Fighting']
        # ['Fighting','Arrest','Vandalism','Assault','Stealing','Arson','NormalVideos','Abuse','Explosion','Robbery','Burglary','Shooting','Shoplifting','RoadAccidents']
        # ['RoadAccidents', 'Assault', 'Vandalism', 'Arrest', 'Shooting', 'NormalVideos', 'Arson', 'Explosion', 'Shoplifting', 'Robbery', 'Stealing', 'Burglary', 'Abuse', 'Fighting']
        op[pred]
        result = op[pred]
        result = 'The predicted output is {}' .format(str(result))
        print(result)
    return render_template('predict.html', text=result)
#     except Exception as e:
#     print("Error : ",str(e))
#     result = "An error occured while processing the image"
# return render_template('predict.html', text=result)


""" Running our application """
if __name__ == "__main__":
    app.run(debug=True)
          