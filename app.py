#=================flask code starts here
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
from werkzeug.utils import secure_filename
from distutils.log import debug
from fileinput import filename
import smtplib 
from email.message import EmailMessage
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
import pickle
import sqlite3
import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer #loading bert sentence model
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
import smote_variants
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import os


UPLOAD_FOLDER = os.path.join('static', 'uploads') 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'welcome'

berts = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

labels = ['Real Job', 'Fraudulent Job']

def getModel():
    extension_cnn2d = Sequential()
    extension_cnn2d.add(Convolution2D(32, (3 , 3), input_shape = (32, 24, 1), activation = 'relu'))
    extension_cnn2d.add(MaxPooling2D(pool_size = (2, 2)))
    extension_cnn2d.add(Convolution2D(32, (3, 3), activation = 'relu'))
    extension_cnn2d.add(MaxPooling2D(pool_size = (2, 2)))
    extension_cnn2d.add(Flatten())
    extension_cnn2d.add(Dense(units = 256, activation = 'relu'))
    extension_cnn2d.add(Dense(units = 2, activation = 'softmax'))
    extension_cnn2d.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    extension_cnn2d.load_weights("model/cnn2d_weights.hdf5")
    return extension_cnn2d



@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/notebook')
def notebook():
    return render_template('FraudJobDetection.html')


@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        f = request.files.get('file')
        data_filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],data_filename))
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'],data_filename)
        data_file_path = session.get('uploaded_data_file_path', None)
        test_data = pd.read_csv(data_file_path,encoding='unicode_escape')
        data = test_data['description'].ravel()
        extension_cnn2d = getModel()
        temp = []
        for i in range(len(data)):
            text = data[i].strip().lower()#loop each test data and then clean by applying NLP techniques 
            text = cleanText(text)
            temp.append(text)
        bert_encode = berts.encode(temp)#apply bert to convert text into numeric embedding
        bert_encode = np.reshape(bert_encode, (bert_encode.shape[0], 32, 24, 1))#reshape data as per CNN2D
        predict = extension_cnn2d.predict(bert_encode)#apply extension to predict job Type
        output = ""
        for i in range(len(predict)):#display each job predicted output            
            pred = np.argmax(predict[i])
            output += "Job Details = "+data[i]+"<br/>"
            output += "Predicted Job Type ====> "+labels[pred]+"<br/><br/>"        
        return render_template('result.html', msg=output)

@app.route('/logon')
def logon():
	return render_template('register.html')

@app.route('/login')
def login():
	return render_template('login.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict1', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("login.html")
    return render_template("register.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("login.html")


    
if __name__ == '__main__':
    app.run()