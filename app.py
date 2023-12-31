import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from keras.models import load_model

NeuralNetwork = load_model('PneuClass(90%).h5')
img_size = 150
  

def predict(name):
    image = st.file_uploader("Upload a file" + name, type=["png", "jpg", "jpeg"])
    if image:
        st.image(image=image)
        im = Image.open(image)
        im.filename = image.name
        SamplePhoto = np.asarray(im)
        resized_arr = cv2.resize(SamplePhoto, (img_size, img_size))

        data = []
        data.append([resized_arr, 0])
        data = np.array(data, dtype = object)
        SamplePhotoXTrain = []
        SamplePhotoYTrain = []
        for feature, label in data:
            SamplePhotoXTrain.append(np.array(feature))
            SamplePhotoYTrain.append(np.array(label))

        SamplePhotoXTrain = np.array(SamplePhotoXTrain) / 255
        SamplePhotoXTrain = SamplePhotoXTrain.reshape(-1, img_size, img_size, 1)
        Prediction = NeuralNetwork.predict(SamplePhotoXTrain)
        FloatNumber = (1.0 - Prediction[0][0]) * 100
        ANS = str("%.2f" % FloatNumber)
        # ANS = str("%.2f" % (1.0 - Prediction[0][0], 3) * 100)   
        if ANS > 60:
          st.header('Пневмония обнаружена\n');
          st.header('Вероятность наличия составляет ' + ANS + '%')

        

def main():
  st.set_page_config(page_title='Диагностирование Пневмонии', page_icon=None, initial_sidebar_state='auto')
  st.title('Диагностирование бактериальной и вирусной пневмонии')
  predict('image')

  

if __name__ == "__main__":
  main()
