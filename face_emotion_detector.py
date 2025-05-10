import cv2
from deepface import DeepFace

def extract_faces_and_emotions(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                results.append(emotion)
                print(f"Frame {frame_count}: {emotion}")
            except Exception as e:
                print(f"Error at frame {frame_count}: {e}")

        frame_count += 1

    cap.release()
    return results
