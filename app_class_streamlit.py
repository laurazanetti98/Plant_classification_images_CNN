import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model


def add_bg_from_url():
    st.markdown(
        f""" <style> .stApp {{ background-image: url( 
        "https://cdn.pixabay.com/photo/2021/03/05/19/50/leaf-6072183_960_720.jpg"); background-attachment: fixed; background-size: cover }} </style> """,
        unsafe_allow_html=True
    )


add_bg_from_url()
st.header('Image Plant Diseases Classification')
st.write("In the following box upload an image:")


def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image / 255, 0)
    prediction = model.predict(image)
    return prediction


new_model = load_model(os.path.join('models', 'image_plant_classifier.h5'))

file = st.file_uploader('Insert image for classification', type=['jpeg', 'jpg', 'bmp', 'png'])
if file is None:
    st.write("Please upload an Image ")
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)

    st.image(test_image, caption="Input Image", width=400)

    pred = predict_class(np.asarray(test_image), new_model)
    class_names = ['colpo_di_fuoco',
                   'flavescenza',
                   'maculatura_bruna',
                   'peronospora',
                   'ticchiolatura']
    result = class_names[np.argmax(pred)]
    output = 'The image is a ' + result
    slot.text('Done')
    st.success(output)
