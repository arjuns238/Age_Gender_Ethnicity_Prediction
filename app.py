#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageFilter
import numpy as np
from tensorflow.keras.models import load_model
#from resizeimage import resizeimage
import matplotlib.pyplot as plt


#Read image
def processImage(filename):
    im = Image.open( './Upload/' + filename ).convert('L')
    #Resizing image code to 24 x 32 
    im = im.resize((48, 48), Image.ANTIALIAS)
    plt.imshow(im, cmap = 'gray')
    im = np.asarray(im, dtype = 'float32')
    im = im.flatten().reshape(1, 48, 48)
    im = im / 255
    return im
    



# In[ ]:


#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from keras.layers import Input
import tarfile


app = Flask(__name__)
 
UPLOAD_FOLDER = './Upload'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')    


@app.route('/', methods=['POST'])
def display_results():
    #Check for valid files
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
    
    picture = processImage(file.filename)
    ethnicity_values_to_labels = ['White','Black','Indian','Asian', 'Other (Hispanic, Latino, Middle Eastern)']
    age_values_to_labels = ['0-20', '21 - 40', '41-60', '61-80', '80+']
    
   
    with tarfile.open('tf-models.tar.gz') as tar:
        Gmodel = load_model('gender_model_new')
        Amodel = load_model('age_model_new')
        Emodel = load_model('ethnicity_model_new')
    
    genderPred = Gmodel.predict(picture)
    print (genderPred)
    agePred = Amodel.predict(picture)
    ethnicityPred = Emodel.predict(picture)
    if genderPred < 0.5:
        gender  = 'Male'
    else:
        gender = 'Female'
    ageIndex = np.argmax(agePred)
    print(genderPred)
    print(agePred)
    print(ethnicityPred)
    ethnicityIndex = np.argmax(ethnicityPred)
    age = age_values_to_labels[ageIndex]
    ethnicity = ethnicity_values_to_labels[ethnicityIndex]
    return render_template('index.html', age = age, gender = gender, ethnicity = ethnicity)





if __name__ == "__main__":
    app.run()

