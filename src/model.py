import tensorflow as tf
from tensorflow.keras import layers

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, units=512, feat_dim=2048):
        super().__init__()
        self.units = units
        self.attention = BahdanauAttention(units)
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.fc1 = layers.Dense(units)
        self.fc2 = layers.Dense(vocab_size)

    def call(self, features, seq):
        context_vector, _ = self.attention(features, tf.zeros((tf.shape(features)[0], self.units)))
        emb = self.embedding(seq)
        x = tf.concat([tf.expand_dims(context_vector, 1), emb], axis=-1)
        output, _, _ = self.lstm(x)
        x = self.fc1(output)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x
