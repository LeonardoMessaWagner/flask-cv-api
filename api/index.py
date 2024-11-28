from flask import Flask, request, jsonify
import os
import cv2
from fer import FER
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "API para Upload e Processamento de Vídeo"

@app.route('/about')
def about():
    return 'About'