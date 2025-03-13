import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

# Load the labels from the labels.txt file
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Custom VideoTransformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        pil_image = Image.fromarray(img)
        processed_image = preprocess_image(pil_image)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions)
        predicted_label = labels[predicted_class]
        print(f"Frame {self.frame_count} - Predicted: {predicted_label}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, predicted_label, (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return img

# Streamlit UI
st.title("Animal Detection")

webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
