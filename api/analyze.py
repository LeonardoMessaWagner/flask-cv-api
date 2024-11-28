import os
import cv2
import json
import tempfile
import mediapipe as mp
from flask import Flask, request, jsonify

# Crie o aplicativo Flask
app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection

def analyze_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    emotions_data = []

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                emotions_data.append({
                    "frame": frame_count,
                    "detection_score": detection.score[0],
                    "bounding_box": detection.location_data.relative_bounding_box.__dict__
                })

        frame_count += 1

    cap.release()
    return emotions_data

# Função para responder à requisição
def handle(request):
    if request.method == 'GET':
        return app.send_static_file('index.html')
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files['video']
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, video.filename)
    video.save(video_path)

    try:
        emotions = analyze_emotions(video_path)
        response = {
            "status": "success",
            "frames_analyzed": len(emotions),
            "emotions_data": emotions
        }
    except Exception as e:
        response = {"status": "error", "message": str(e)}

    os.remove(video_path)
    return jsonify(response)

# Adicione a função para servir no Vercel
if __name__ == '__main__':
    app.run(debug=True)
