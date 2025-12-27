import os
import string
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer


# ===== Load and clean captions =====
def load_captions(captions_file):
    captions = {}

    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if lines[0].lower().startswith("image"):
        lines = lines[2:]

    for i in range(0, len(lines), 2):
        img_id = lines[i].strip()
        if i + 1 < len(lines):
            caption = lines[i + 1].strip()
            captions.setdefault(img_id, []).append(caption)

    print(f"✅ Loaded {len(captions)} image captions.")
    return captions


def clean_caption(c):
    c = c.lower()
    c = c.replace('-', ' ')
    c = c.translate(str.maketrans('', '', string.punctuation))
    c = ' '.join([w for w in c.split() if w.isalpha()])
    return 'startseq ' + c + ' endseq'


def build_tokenizer(captions_dict, num_words=8000):
    all_caps = []
    for caps in captions_dict.values():
        for c in caps:
            all_caps.append(clean_caption(c))

    tok = Tokenizer(num_words=num_words, oov_token='unk')
    tok.fit_on_texts(all_caps)
    print(f"✅ Tokenizer built. Total unique words: {len(tok.word_index)}")
    return tok


# ===== Extract features using InceptionV3 =====
def extract_features(image_dir, image_list):
    features = {}
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    print(f"\n🔍 Checking image directory: {image_dir}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"❌ Directory not found: {image_dir}")

    for img_name in tqdm(image_list, desc='Extracting features'):
        path = os.path.join(image_dir, img_name)

        if not os.path.exists(path):
            print(f"⚠️ Missing image: {path}")
            continue

        try:
            img = Image.open(path).convert('RGB').resize((299, 299))
            img = np.array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            feat = model.predict(img, verbose=0)
            features[img_name] = feat[0]
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    print(f"\n✅ Extracted features for {len(features)} images.")
    return features


if __name__ == "__main__":
    print("🚀 Starting Preprocessing Script...")

    captions_path = "../data/Flickr8k_text/Flickr8k.token.txt"
    images_dir = "../data/Flickr8k_images/Images"
    save_dir = "../data"

    # Load and process captions
    captions_dict = load_captions(captions_path)
    image_list = list(captions_dict.keys())

    # Clean all captions
    for img_id, caps in captions_dict.items():
        captions_dict[img_id] = [clean_caption(c) for c in caps]

    # Extract image features
    features = extract_features(images_dir, image_list)

    # Build tokenizer
    tokenizer = build_tokenizer(captions_dict)

    # Save all to .pkl files
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(features, open(os.path.join(save_dir, "features.pkl"), "wb"))
    pickle.dump(captions_dict, open(os.path.join(save_dir, "captions.pkl"), "wb"))
    pickle.dump(tokenizer, open(os.path.join(save_dir, "tokenizer.pkl"), "wb"))

    print("\n💾 Saved files:")
    print(f"📁 {save_dir}/features.pkl")
    print(f"📁 {save_dir}/captions.pkl")
    print(f"📁 {save_dir}/tokenizer.pkl")
    print("✅ Preprocessing complete and data saved successfully!")
