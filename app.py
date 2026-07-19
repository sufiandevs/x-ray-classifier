import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import json
import matplotlib.pyplot as plt
import base64

# --- Configuration --- #
MODEL_PATH = "cnn_xray_model_weights.weights.h5"  # Only weights file
GOOGLE_DRIVE_FILE_ID = "188XLpeOEW9VmMb8Hl_7GniqeGbB__zHQ"  # <-- UPDATE THIS
IMG_SIZE = (150, 150)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
HISTORY_FILE = "training_history.json"
LOGO_PATH = "deep_vison logo.jpg"


# --- Build Model Architecture (must match training exactly) --- #
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Model Loading --- #
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.write("Downloading model weights from Google Drive...")
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("Model weights downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model weights: {str(e)}")
            st.stop()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.error(f"The model weights file {MODEL_PATH} is missing or empty.")
        st.stop()

    st.write("Loading model...")
    try:
        model = create_model()
        model.load_weights(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


# --- Prediction Function --- #
def predict_image(model, image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    probability_pneumonia = predictions[0][0]

    if probability_pneumonia >= 0.5:
        predicted_class_idx = 1
        actual_confidence = probability_pneumonia
    else:
        predicted_class_idx = 0
        actual_confidence = 1 - probability_pneumonia

    label = CLASS_NAMES[predicted_class_idx]
    confidence_percent = round(float(actual_confidence) * 100, 2)

    return label, confidence_percent


# --- Custom CSS --- #
st.markdown("""
<style>
@keyframes rotate3D {
    from { transform: rotateY(0deg); }
    to { transform: rotateY(360deg); }
}
.animated-logo {
    animation: rotate3D 5s infinite linear;
    perspective: 1000px;
    display: block;
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


# --- Streamlit UI --- #
st.set_page_config(page_title="X-Ray Image Classifier", page_icon="🩻", layout="wide")

# Logo
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{encoded_string}" class="animated-logo" width="150">
        <p style="text-align: center;">Deep Vision Logo</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning(f"Company logo not found at {LOGO_PATH}")

st.title("Pneumonia X-Ray Image Classifier")
st.write("Upload a chest X-ray image to get a prediction (Normal or Pneumonia).")

model = get_model()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("")

        with st.spinner('Classifying image...'):
            label, confidence = predict_image(model, image)

        st.success(f"Prediction: **{label}** (Confidence: **{confidence:.2f}%**)")

        if label == "PNEUMONIA":
            st.warning("Disclaimer: This is an AI-based prediction and should not be used as a substitute for professional medical advice.")

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("---")

# Training graphs
if st.button("Show Training Accuracy and Loss Graphs"):
    st.subheader("Training Progress")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history_data = json.load(f)

        epochs_range = range(len(history_data['accuracy']))

        fig_train, axes_train = plt.subplots(1, 2, figsize=(15, 6))

        axes_train[0].plot(epochs_range, history_data['accuracy'], label='Training Accuracy', color='blue')
        axes_train[0].legend(loc='lower right')
        axes_train[0].set_title('Training Accuracy')
        axes_train[0].set_xlabel('Epoch')
        axes_train[0].set_ylabel('Accuracy')
        axes_train[0].grid(True)

        axes_train[1].plot(epochs_range, history_data['loss'], label='Training Loss', color='red')
        axes_train[1].legend(loc='upper right')
        axes_train[1].set_title('Training Loss')
        axes_train[1].set_xlabel('Epoch')
        axes_train[1].set_ylabel('Loss')
        axes_train[1].grid(True)

        st.pyplot(fig_train)
    else:
        st.warning("Training history not found.")
        st.info(f"Expected history file at: {HISTORY_FILE}")

st.caption("Model developed using TensorFlow/Keras. For educational purposes only.")
