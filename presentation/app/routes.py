from app import app 
from flask import render_template, send_from_directory, request
import os
import base64
import numpy as np
import sys
from io import BytesIO
from PIL import Image

BASE_URL = os.path.abspath(os.path.dirname(__file__))
MODULES_APP_FOLDER = os.path.join(BASE_URL, "node_modules")
ANGULAR_APP_FOLDER = os.path.join(BASE_URL, "static/js/app")
sys.path.append(os.path.join(BASE_URL, "../../utils"))

import readImage as read



@app.route('/node_modules/<path:filename>')
def modules_app_folder(filename):
    return send_from_directory(MODULES_APP_FOLDER, filename)

@app.route('/angular/<path:filename>')
def angular_app_folder(filename):
    return send_from_directory(ANGULAR_APP_FOLDER, filename)


@app.route('/')
@app.route('/index')

def index():        
    return render_template('index.html')

@app.route('/image', methods=["GET","POST"])
def image():
    id = request.args.get('id')
    img = read.getImage('../noPooling/results/results', int(id))
    buff = BytesIO()
    img.save(buff, format="PNG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    # print('new string', new_image_string)
    return new_image_string