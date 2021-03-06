from flask import Flask, url_for, request, render_template, Response, redirect
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from io import BytesIO
from os import system

app = Flask(__name__)

modelPath = 'path\\models'
model = load_model(f'{modelPath}\\HistoCNN.model')
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
			model = load_model(f'{modelPath}\\HistoCNN.model')
			typ = 'h'
		if request.form.get('Blood Cell'):
			model = load_model(f'{modelPath}\\BloodNN.model')
			typ = 'b'
		if request.form.get('Leukemia'):
			model = load_model(f'{modelPath}\\LeukNN.model')
			typ = 'l'
	return ''

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		global model

		files = request.files.getlist('file')
		preds = []
		size = len(files)

		if str(files[0]) == "<FileStorage: '' ('application/octet-stream')>":
			return render_template('index.html', output='<br><p style="text-align: center">NO FILES DETECTED.</p><br><p style="text-align: center">Please select images before uploading.</p><br><p style="text-align: center">Supported filetypes are: JPG, PNG, TIF, BMP...</p>')

		try:
			for row in range(size):
				preds.append(network(fix(files[row]), model)[0])
				system('cls')
				print(f'Processing files: {row+1}/{size} - {int((row+1)/size*100)}%')
			return render_template('index.html', output=send(preds))
		except:
			return render_template('index.html', output='<br><p style="text-align: center">UNSUPPORTED FILETYPE</p><br><p style="text-align: center">Please upload only the following filetypes: JPG, PNG, TIF, BMP...</p>')
	return None

if __name__ == '__main__':
	system('cls')
	app.run(debug=True)	
