import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint


# ===== Load preprocessed data =====
def load_data():
    data_dir = os.path.join("..", "data")
    with open(os.path.join(data_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    with open(os.path.join(data_dir, "captions.pkl"), "rb") as f:
        captions = pickle.load(f)
    with open(os.path.join(data_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    # Normalize keys to ensure match between captions and features
    features = {os.path.basename(k).split('.')[0]: v for k, v in features.items()}
    captions = {os.path.basename(k).split('.')[0]: v for k, v in captions.items()}

    print(f"✅ Loaded {len(features)} image features.")
    print(f"✅ Loaded {len(captions)} image captions.")
    return features, captions, tokenizer


# ===== Prepare training sequences =====
def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1

    for key, desc_list in descriptions.items():
        photo = photos.get(key)
        if photo is None:
            continue

        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq_index = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = np.zeros(vocab_size)
                out_seq[out_seq_index] = 1.0
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)

    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)

    print(f"✅ Training samples: {len(X1)}")
    return X1, X2, y


# ===== Define model =====
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# ===== Training =====
def train():
    features, captions, tokenizer = load_data()

    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for desc in captions.values() for c in desc)

    print(f"🧠 Vocabulary Size: {vocab_size}")
    print(f"🕓 Max Caption Length: {max_length}")

    X1, X2, y = create_sequences(tokenizer, max_length, captions, features)

    model = define_model(vocab_size, max_length)
    model.summary()

    checkpoint = ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True)
    model.fit([X1, X2], y, epochs=10, batch_size=64, verbose=1, callbacks=[checkpoint])

    print("✅ Model training complete and saved as model.h5")


# ===== Run script =====
if __name__ == "__main__":
    print("🚀 Starting Training Script...")
    train()
