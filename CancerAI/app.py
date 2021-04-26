from flask import Flask, url_for, request, render_template, Response, redirect
from jinja2 import Template
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

print('\nSite loaded. Check http://127.0.0.1:5000/\n')

modelPath = 'EnterYourPathHere\\CancerAI\\models'
model = tf.keras.models.load_model(f'{modelPath}\\HistoCNN.model')
typ = 'h'

def fix(img):
	mem = BytesIO()
	img.save(mem)
	data = np.fromstring(mem.getvalue(), dtype=np.uint8)
	img = cv2.imdecode(data, 1)

	return img[:,:,::-1]

def network(img, model):
	inputs = np.array(cv2.resize(img, (96, 96)))/255.0
	return model.predict(np.array([inputs]))

def color(pred):
	return f'style="color: rgb({255-	pred*2.55},{pred*2.55},0);"'

def send(pred):
	global typ
	table = ''

	size = len(pred)
	if typ == 'h':
		scores = sum(pred)*100/size

		table += '<thead><tr>'
		table += '<th>Benign</th>'
		table += '<th>Malignant</th>'
		table += '</tr></thead>'

		table += '<tbody class="table-hover"><tr>'
		table += f'<td {color(int(100-scores))}>{int(100-scores)}%</td>'
		table += f'<td {color(int(scores))}>{int(scores)}%</td>'
		table += '</tr></tbody>'

	elif typ == 'l':
		scores = sum(pred)*100/size

		table += '<thead><tr>'
		table += '<th>Normal</th>'
		table += '<th>Leukemia</th>'
		table += '</tr></thead>'

		table += '<tbody class="table-hover"><tr>'
		table += f'<td {color(int(100-scores))}>{int(100-scores)}%</td>'
		table += f'<td {color(int(scores))}>{int(scores)}%</td>'
		table += '</tr></tbody>'

	elif typ == 'b':
		pred = np.sum(pred, axis=0)

		table += '<thead><tr>'
		table += '<th>Eosinophil</th>'
		table += '<th>Lymphocyte</th>'
		table += '<th>Monocyte</th>'
		table += '<th>Neutrophil</th>'
		table += '</tr></thead>'

		table += '<tbody class="table-hover"><tr>'
		table += f'<td {color(int(pred[0]*100/size))}>{int(pred[0]*100/size)}%</td>'
		table += f'<td {color(int(pred[1]*100/size))}>{int(pred[1]*100/size)}%</td>'
		table += f'<td {color(int(pred[2]*100/size))}>{int(pred[2]*100/size)}%</td>'
		table += f'<td {color(int(pred[3]*100/size))}>{int(pred[3]*100/size)}%</td>'
		table += '</tr></tbody>'
	
	return '<table style="margin-left:auto;margin-right:auto">' + table + '</table>'

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
		size = len(files)
		for row in range(size):
			preds.append(network(fix(files[row]), model)[0])

		return render_template('index.html', output=send(preds))
	return None

if __name__ == '__main__':
	app.run(debug=True)

'''
To-Do List
	Progress Bar
'''
