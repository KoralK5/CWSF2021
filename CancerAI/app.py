import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

modelPath = 'D:\\Users\\Koral Kulacoglu\\Coding\\python\\AI\\CWSF\\CancerAI\\models'
model = tf.keras.models.load_model(f'{modelPath}\\HistoCNN.model')
typ = 'h'

print('\nModel loaded. Check http://127.0.0.1:5000/\n')

def fix(img):
	mem = BytesIO()
	img.save(mem)
	data = np.fromstring(mem.getvalue(), dtype=np.uint8)
	img = cv2.imdecode(data, 1)

	return img[:,:,::-1]

def network(img, model):
	inputs = np.array(cv2.resize(img, (96, 96)))/255.0
	return model.predict(np.array([inputs]))

def send(pred):
	global typ

	msg = '<h3>DETECTIONS</h3>'
	size = len(pred)
	if typ == 'h':
		scores = sum(pred)*100/size
		msg += '<br><p>Benign:' + f' {int(100-scores)}%</p>'
		msg += '<br><p>Malignant:' + f' {int(scores)}%</p>'
		
	elif typ == 'l':
		scores = sum(pred)*100/size
		msg += '<br><p>Normal:' + f' {int(100-scores)}%</p>'
		msg += '<br><p>Leukemia:' + f' {int(scores)}%</p>'

	elif typ == 'b':
		pred = np.sum(pred, axis=0)
		msg += '<br><p>Eosinophil:' + f' {int(pred[0]*100/size)}%</p>'
		msg += '<br><p>Lymphocyte:' + f' {int(pred[1]*100/size)}%</p>'
		msg += '<br><p>Monocyte:' + f' {int(pred[2]*100/size)}%</p>'
		msg += '<br><p>Neutrophil:' + f' {int(pred[3]*100/size)}%</p>'
	
	return '<div>'+msg+'</div>'

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
	if request.method == 'POST':
		global model, typ
		
		if request.form.get('Lymph Node'):
			model = tf.keras.models.load_model(f'{modelPath}\\HistoCNN.model')
			typ = 'h'
		if request.form.get('Blood Cell'):
			model = tf.keras.models.load_model(f'{modelPath}\\BloodNN.model')
			typ = 'b'
		if request.form.get('Leukemia'):
			model = tf.keras.models.load_model(f'{modelPath}\\LeukNN.model')
			typ = 'l'
	return ''

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		global model

		files, preds = request.files.getlist('file'), []
		for row in files:
			preds.append(network(fix(row), model)[0])
		msg = send(preds)
		
		print(msg)

		return render_template('index.html', output=msg)
	return None

if __name__ == '__main__':
	app.run(debug=True)

'''
To-Do List
	Progress Bar
	Visible Results
	Descriptions On Left
	Results On Right
'''
