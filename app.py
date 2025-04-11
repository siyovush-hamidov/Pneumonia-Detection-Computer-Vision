import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
import os

# Загрузка модели
NeuralNetwork = load_model('model/PneuClass(90%).h5')
img_size = 150

# Проверяем наличие файлов
assert os.path.exists('xray_samples/Healthy.jpg'), "Файл Healthy.jpg не найден."
assert os.path.exists('xray_samples/Pneumonia.jpeg'), "Файл Pneumonia.jpeg не найден."

def resize_image(img_path, max_width=400):
    """Уменьшает изображение для отображения."""
    img = Image.open(img_path)
    if img.width > max_width:
        scale = max_width / img.width
        new_size = (max_width, int(img.height * scale))
        img = img.resize(new_size, Image.ANTIALIAS)
    return img

def predict(name):
    """Функция загрузки и предсказания по изображению."""
    image = st.file_uploader("Аксро бор кунед: " + name, type=["png", "jpg", "jpeg"])
    if image:
        im = Image.open(image)
        SamplePhoto = np.asarray(im)
        resized_arr = cv2.resize(SamplePhoto, (img_size, img_size))

        # Подготовка данных
        data = np.array([resized_arr], dtype=np.float32) / 255
        data = data.reshape(-1, img_size, img_size, 1)

        try:
            # Предсказание
            Prediction = NeuralNetwork.predict(data)
            FloatNumber = (1.0 - Prediction[0][0]) * 100
            ANS = str("%.2f" % FloatNumber)

            if FloatNumber > 60:
                st.markdown("<h4 style='text-align: center; color: red;'>Аломатҳои бемории илтиҳоби шуш мушоҳида мешаванд.</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: red;'>Эҳтимолияти беморӣ: {ANS}%</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='text-align: center; color: green;'>Аломатҳои беморӣ мушоҳида намешаванд.</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: green;'>Эҳтимолияти беморӣ: {ANS}%</h5>", unsafe_allow_html=True)

            st.image(image=image, caption='Аксҳои рентгении таҳлилшуда', use_column_width=True)

        except Exception as e:
            st.error(f"Хатогӣ ҳангоми истифодаи модел: {str(e)}")

def main():
    """Саҳифаи асосии Streamlit."""
    st.markdown("<h2 style='text-align: center; color: white;'>Модели омӯзиши мошинӣ барои ташхиси илтиҳоби шуш</h2>", unsafe_allow_html=True)
    st.image(
        'miscellaneous/attention.webp',
        caption='“Саломатӣ беҳтар аз ҳама неъматҳост. —Артур Шопенгауэр”',
        use_column_width=True
    )
    st.markdown("<h3 style='text-align: center; color: white;'>Модел чӣ гуна кор мекунад?</h3>", unsafe_allow_html=True)
    st.write("Истифодабаранда бояд акси рентгениро бор кунад ва модел натиҷаи эҳтимолияти бемориро нишон медиҳад.")

    # Намоиши мисолҳо
    col1, col2 = st.columns(2)
    col1.image(resize_image('xray_samples/Healthy.jpg'), caption='Синаи шахси солим')
    col2.image(resize_image('xray_samples/Pneumonia.jpeg'), caption='Синаи шахси бемор')

    # Боргузорӣ ва пешгӯӣ
    st.markdown("<h5 style='text-align: center; color: white;'>Аксро бор кунед</h5>", unsafe_allow_html=True)
    predict('акс')  # Номи оддии тасвир барои боркунӣ

if __name__ == "__main__":
    main()
