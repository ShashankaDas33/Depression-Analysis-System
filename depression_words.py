import pandas as pd

# Load depression-related words
depression_words = pd.read_csv("depression_words.csv")['word'].tolist()

# Read transcribed text from a file
with open("transcription.txt", "r", encoding="utf-8") as f:
    transcribed_text = f.read()

# Preprocess text: lowercasing and splitting
words_in_text = transcribed_text.lower().split()

# Find matches
matched_words = [word for word in words_in_text if word in depression_words]

# Results
print(f"Matched Depression-related Words: {matched_words}")
print(f"Number of Depression-related Words Detected: {len(matched_words)}")
