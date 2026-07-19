import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import json
import matplotlib.pyplot as plt
import base64 # Import base64 for embedding logo

# --- Configuration --- #
MODEL_PATH = "cnn_xray_model_v2.keras"
# IMPORTANT: Update this GOOGLE_DRIVE_FILE_ID with the ID of your 'cnn_xray_model_v2.keras' file.
# Make sure the file is publicly accessible or appropriately shared.
# To get the ID: Go to Google Drive, right-click on your file, select "Share", and copy the link.
# The ID is the part after 'id=' in the URL.
GOOGLE_DRIVE_FILE_ID = "1FnV4ptDjiyE7ugLTfAsvYEkwMnMWpCYu" # <--- USER MUST UPDATE THIS
IMG_SIZE = (150, 150)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
HISTORY_FILE = "training_history.json" # Relative path, assuming app.py is in the same directory as this file
LOGO_PATH = "deep_vison logo.jpg" # Absolute path to the company logo

# --- Model Loading (with caching for Streamlit) --- #
@st.cache_resource
def get_model():
    """
    Downloads the model from Google Drive if not present and loads it.
    Uses Streamlit's caching mechanism to load the model only once.
    """
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.write("Downloading model from Google Drive...")
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH) # Clean up potentially corrupted/empty file
            # Ensure gdown is installed in the environment where Streamlit runs
            if GOOGLE_DRIVE_FILE_ID == "YOUR_NEW_GOOGLE_DRIVE_FILE_ID_HERE":
                st.error("ERROR: Please update GOOGLE_DRIVE_FILE_ID in app.py with your model's public Google Drive ID.")
                st.stop()
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.stop()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.error(f"The model file {MODEL_PATH} is missing or empty after download attempt.")
        st.stop()

    st.write("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Please check the model file integrity and ensure it was saved with Keras 3.")
        st.stop()

# --- Prediction Function --- #
def predict_image(model, image):
    """
    Preprocesses the image and makes a prediction using the Keras model.
    """
    # Resize image to target size
    image = image.resize(IMG_SIZE)
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)

    # Process binary classification (sigmoid output)
    probability_pneumonia = predictions[0][0]

    if probability_pneumonia >= 0.5:
        predicted_class_idx = 1 # PNEUMONIA
        actual_confidence = probability_pneumonia
    else:
        predicted_class_idx = 0 # NORMAL
        actual_confidence = 1 - probability_pneumonia

    label = CLASS_NAMES[predicted_class_idx]
    confidence_percent = round(float(actual_confidence) * 100, 2)

    return label, confidence_percent

# --- Custom CSS for Animation --- 
st.markdown("""
<style>
@keyframes rotate3D {
    from {
        transform: rotateY(0deg);
    }
    to {
        transform: rotateY(360deg);
    }
}
.animated-logo {
    animation: rotate3D 5s infinite linear;
    perspective: 1000px; /* Add perspective for 3D effect */
    display: block; /* Center image within its container */
    margin-left: auto;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


# --- Streamlit UI --- #
st.set_page_config(page_title="X-Ray Image Classifier", page_icon="レントゲン", layout="wide")

# Company Logo with Animation
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
    st.warning(f"Company logo not found at {LOGO_PATH}. Please ensure the path is correct.")

st.title("Pneumonia X-Ray Image Classifier")
st.write("Upload a chest X-ray image to get a prediction (Normal or Pneumonia).")

# Load the model
model = get_model()

# Image Uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Make prediction
        with st.spinner('Classifying image...'):
            label, confidence = predict_image(model, image)

        st.success(f"Prediction: **{label}** (Confidence: **{confidence:.2f}%**)")

        if label == "PNEUMONIA":
            st.warning("Disclaimer: This is an AI-based prediction and should not be used as a substitute for professional medical advice. Please consult a doctor for diagnosis.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.write("Please try uploading a valid image file.")

st.markdown("---") # Separator

# Button to show training graphs
if st.button("Show Training Accuracy and Loss Graphs"):
    st.subheader("Training Progress")
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history_data = json.load(f)

        epochs_range = range(len(history_data['accuracy']))

        fig_train, axes_train = plt.subplots(1, 2, figsize=(15, 6)) # Create a new figure for these plots

        axes_train[0].plot(epochs_range, history_data['accuracy'], label='Training Accuracy')
        axes_train[0].legend(loc='lower right')
        axes_train[0].set_title('Training Accuracy')
        axes_train[0].set_xlabel('Epoch')
        axes_train[0].set_ylabel('Accuracy')
        axes_train[0].grid(True)

        axes_train[1].plot(epochs_range, history_data['loss'], label='Training Loss')
        axes_train[1].legend(loc='upper right')
        axes_train[1].set_title('Training Loss')
        axes_train[1].set_xlabel('Epoch')
        axes_train[1].set_ylabel('Loss')
        axes_train[1].grid(True)

        st.pyplot(fig_train) # Display the figure
    else:
        st.warning("Training history not found. Please ensure the model was trained and history saved.")
        st.info(f"Expected history file at: {HISTORY_FILE}")


st.caption("Model developed using TensorFlow/Keras. For educational purposes only.")
