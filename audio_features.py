import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import os

def extract_audio_from_video(video_path, audio_path="temp.wav"):
    try:
        # Extract audio from the video and save it as a .wav file
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_features(audio_path):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(audio_path)

        # Extract pitch using YIN algorithm
        pitch = librosa.yin(y, fmin=50, fmax=300)
        avg_pitch = np.mean(pitch)

        # Calculate the energy (root mean square)
        energy = np.mean(librosa.feature.rms(y=y))

        # Detect the tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {
            "avg_pitch": float(avg_pitch),
            "energy": float(energy),
            "tempo": float(tempo)
        }
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

# Example usage (commented out to avoid execution when imported)
"""
if __name__ == "__main__":
    # Path to the video file
    video_path = "model_training_video.mp4"
    
    if os.path.exists(video_path):
        # Extract audio from the video
        audio_path = extract_audio_from_video(video_path)
        
        if audio_path:
            # Extract audio features
            audio_features = extract_audio_features(audio_path)
            
            # If audio features were extracted, print them
            if audio_features:
                print(f"Average Pitch: {audio_features['avg_pitch']}")
                print(f"Energy: {audio_features['energy']}")
                print(f"Tempo: {audio_features['tempo']}")
    else:
        print(f"Video file not found: {video_path}")
"""
