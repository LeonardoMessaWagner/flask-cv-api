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

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"erro": "Nenhum arquivo de vídeo foi enviado"}), 400

    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    # Processar o vídeo e detectar emoções
    emocao_por_segundo = detectar_emocoes_no_video(video_path)
    
    return jsonify(emocao_por_segundo), 200

def detectar_emocoes_no_video(video_path):
    # Iniciar o detector de emoções
    detector = FER()
    captura = cv2.VideoCapture(video_path)
    
    emocao_por_segundo = []
    segundo_atual = 0
    fps = captura.get(cv2.CAP_PROP_FPS)

    while captura.isOpened():
        ret, frame = captura.read()
        if not ret:
            break
        
        # Processar um frame a cada segundo
        if int(captura.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps) == 0:
            resultados_emocao = detector.detect_emotions(frame)
            if resultados_emocao:
                emocao = resultados_emocao[0]['emotions']
                emocao_dominante = max(emocao, key=emocao.get)
                emocao_por_segundo.append({
                    'segundo': segundo_atual,
                    'emocao': emocao_dominante,
                    'pontuacao': emocao[emocao_dominante]
                })
            segundo_atual += 1

    captura.release()
    return emocao_por_segundo

if __name__ == "__main__":
    app.run(debug=True)