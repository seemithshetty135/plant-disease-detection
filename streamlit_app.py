
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("🌿 Plant Disease Detection (ResNet)")

model = load_model("model.h5")

uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:
    img = image.load_img(uploaded, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    st.image(uploaded)
    st.write("Prediction:", preds)
