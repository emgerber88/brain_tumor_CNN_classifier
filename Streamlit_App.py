#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import streamlit as st

# load model
model = keras.models.load_model('/Users/eli/Desktop/brain_tumor_CNN_classifier/final_model')

# display streamlit page title and prompt
st.title('What type of brain tumor is this?')
st.write("Upload an image from an MRI below and we'll tell you what type of brain tumor the patient has.")
image_input = st.file_uploader('Upload your image here:')

# preprocess input, vectorize, make a prediction, and display result
button = st.button('Classify Image')

if button:
    # generate image data
    img = load_img(image_input, target_size = (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # add an extra dimension to make it rank 4
    datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=keras.applications.vgg16.preprocess_input)
    generator = datagen.flow(x = img)
    pred = model.predict(generator)
    if np.argmax(pred, axis=1) == 0:
        st.write('This tumor appears to be a glioma.')
    elif np.argmax(pred, axis=1) == 1:
        st.write('This tumor appears to be a meningioma.')
    elif np.argmax(pred, axis=1) == 2:
        st.write('There does not appear to be a tumor present.')
    elif np.argmax(pred, axis=1) == 3:
        st.write('This tumor appears to be an other tumor type.')
    else:
        st.write('This tumor appears to be a pituitary tumor.')