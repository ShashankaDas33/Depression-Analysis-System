from face_emotion_detector import extract_faces_and_emotions
from audio_features import extract_audio_from_video, extract_audio_features
import whisper
import pandas as pd
import os

def analyze_depression(video_path):
    try:
        # Step 1: Facial Emotion Analysis
        emotions = extract_faces_and_emotions(video_path)
        if not emotions:
            print("Warning: No emotions detected in the video")
            depression_score_face = 0
        else:
            # Scoring emotions
            depression_weighted_emotions = ['sad', 'angry', 'neutral']
            depression_score_face = sum(1 for e in emotions if e in depression_weighted_emotions) / len(emotions)

        # Step 2: Audio Analysis
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            print("Warning: Could not extract audio from video")
            depression_score_audio = 0
        else:
            audio_features = extract_audio_features(audio_path)
            if not audio_features:
                print("Warning: Could not extract audio features")
                depression_score_audio = 0
            else:
                # Scoring audio: lower pitch + low energy + slow tempo = more depressive
                score_audio = 0
                if audio_features['avg_pitch'] < 130:  # in Hz
                    score_audio += 1
                if audio_features['energy'] < 0.01:
                    score_audio += 1
                if audio_features['tempo'] < 90:
                    score_audio += 1
                depression_score_audio = score_audio / 3  # normalize

        # Step 3: Text Analysis
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            transcribed_text = result['text']

            # Save transcription
            with open("transcription.txt", "w", encoding="utf-8") as f:
                f.write(transcribed_text)

            # Load depression-related words
            if not os.path.exists("depression_words.csv"):
                print("Warning: depression_words.csv not found")
                depression_score_text = 0
                matched_words = []
            else:
                depression_words = pd.read_csv("depression_words.csv")['word'].tolist()
                words_in_text = transcribed_text.lower().split()
                matched_words = [word for word in words_in_text if word in depression_words]
                depression_score_text = len(matched_words) / len(depression_words)  # normalize
        except Exception as e:
            print(f"Warning: Text analysis failed: {e}")
            depression_score_text = 0
            matched_words = []

        # Step 4: Final Depression Likelihood Score
        final_score = round((0.5 * depression_score_face + 0.3 * depression_score_audio + 0.2 * depression_score_text) * 100, 2)

        # Return all results
        return {
            'depression_score_face': depression_score_face,
            'depression_score_audio': depression_score_audio,
            'depression_score_text': depression_score_text,
            'final_score': final_score,
            'matched_words': matched_words,
            'transcribed_text': transcribed_text if 'transcribed_text' in locals() else "",
            'emotions': emotions if emotions else []
        }

    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    # Use relative path or allow user to specify video path
    video_path = "model_training_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
    else:
        results = analyze_depression(video_path)
        if results:
            print("\n🔍 Depression Likelihood Analysis:")
            print(f"Facial Emotion Score: {results['depression_score_face']:.2f}")
            print(f"Audio Feature Score: {results['depression_score_audio']:.2f}")
            print(f"Text Feature Score: {results['depression_score_text']:.2f}")
            print(f"Matched Depression-related Words: {results['matched_words']}")
            print(f"📊 Final Depression Likelihood Score: {results['final_score']}%")
