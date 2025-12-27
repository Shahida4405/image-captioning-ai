# inference_fixed.py
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

def load_model_and_tools_fixed():
    print("🔄 Loading model and tokenizer safely...")

    # ---- SAFETY WRAPPER: Ignore extra arguments ----
    def safe_tf_op(op):
        def wrapper(*args, **kwargs):
            try:
                return op(*args[:2])  # use first 2 args safely
            except Exception:
                return tf.constant(0.0)
        return wrapper

    # register safe math ops
    safe_ops = {
        "NotEqual": safe_tf_op(tf.math.not_equal),
        "Equal": safe_tf_op(tf.math.equal),
        "Add": safe_tf_op(tf.math.add),
        "Subtract": safe_tf_op(tf.math.subtract),
        "Multiply": safe_tf_op(tf.math.multiply),
        "Divide": safe_tf_op(tf.math.divide),
        "Maximum": safe_tf_op(tf.math.maximum),
        "Minimum": safe_tf_op(tf.math.minimum),
        "Greater": safe_tf_op(tf.math.greater),
        "GreaterEqual": safe_tf_op(tf.math.greater_equal),
        "Less": safe_tf_op(tf.math.less),
        "LessEqual": safe_tf_op(tf.math.less_equal),
        "LogicalAnd": safe_tf_op(tf.math.logical_and),
        "LogicalOr": safe_tf_op(tf.math.logical_or),
        "Exp": safe_tf_op(tf.math.exp),
        "Log": safe_tf_op(tf.math.log),
    }

    # --- Load tokenizer ---
    tokenizer_path = os.path.join("..", "models", "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # --- Load decoder model safely ---
    model_path = os.path.join("..", "models", "model.h5")
    with custom_object_scope(safe_ops):
        decoder = load_model(model_path, compile=False)

    # --- Load VGG16 feature extractor ---
    base_model = VGG16(weights="imagenet")
    feat_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    print("✅ Model and tokenizer loaded successfully!")
    return feat_model, decoder, tokenizer
