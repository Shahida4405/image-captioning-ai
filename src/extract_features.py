import os
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def extract_features(image_dir):
    model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    features = {}

    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(image_dir, img_name)
        try:
            image = Image.open(path).convert("RGB").resize((299, 299))
            image = np.expand_dims(np.array(image), axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img_name] = feature[0]
        except Exception as e:
            print(f"⚠️ Error with {img_name}: {e}")

    print(f"\n✅ Extracted features for {len(features)} images.")
    with open("../data/features.pkl", "wb") as f:
        pickle.dump(features, f)
    print("✅ Saved features.pkl")

if __name__ == "__main__":
    image_dir = r"..\data\Flickr8k_images\Images"  # ✅ Windows path
    if not os.path.exists(image_dir):
        print("❌ Image folder not found!")
    else:
        extract_features(image_dir)
