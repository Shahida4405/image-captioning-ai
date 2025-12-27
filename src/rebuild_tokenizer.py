# src/rebuild_tokenizer.py
import pickle
from keras.preprocessing.text import Tokenizer

# Path to captions file
captions_path = "../data/Flickr8k_text/Flickr8k.token.txt"

# Read all captions
with open(captions_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Extract only caption text (not image IDs)
all_captions = []
for line in lines:
    caption = line.strip().split("\t")[1]
    all_captions.append(caption)

# Build tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

# Save tokenizer
with open("../model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer rebuilt and saved to model/tokenizer.pkl")
