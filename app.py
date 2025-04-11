import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import os

# Кэширование загрузки модели для увеличения скорости
@st.cache_resource
def get_model():
    try:
        model = tf.keras.models.load_model('model/PneuClass(90%).h5')
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {str(e)}")
        return None

# Загрузка модели с использованием кэша
NeuralNetwork = get_model()
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
        # Заменяем устаревший Image.ANTIALIAS на Image.LANCZOS
        img = img.resize(new_size, Image.LANCZOS)
    return img

def predict(name):
    """Функция загрузки и предсказания по изображению."""
    image = st.file_uploader("Расм бор кунед: " + name, type=["png", "jpg", "jpeg"])
    if image:
        im = Image.open(image)
        SamplePhoto = np.asarray(im)
        
        # Проверка, является ли изображение цветным или черно-белым
        if len(SamplePhoto.shape) == 3 and SamplePhoto.shape[2] == 3:
            # Конвертируем RGB изображение в оттенки серого
            SamplePhoto = cv2.cvtColor(SamplePhoto, cv2.COLOR_RGB2GRAY)
            
        resized_arr = cv2.resize(SamplePhoto, (img_size, img_size))
        
        # Подготовка данных
        data = np.array([resized_arr], dtype=np.float32) / 255
        data = data.reshape(-1, img_size, img_size, 1)
        
        try:
            if NeuralNetwork is not None:
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
                
                st.image(image=image, caption='Расмҳои рентгении таҳлилшуда', use_column_width=True)
            else:
                st.error("Модель не загружена. Пожалуйста, проверьте наличие файла модели.")
        except Exception as e:
            st.error(f"Хатогӣ ҳангоми истифодаи модел: {str(e)}")

def main():
    """Саҳифаи асосии Streamlit."""
    st.markdown("<h2 style='text-align: center; color: white;'>Модели омӯзиши мошинӣ барои ташхиси илтиҳоби шуш</h2>", unsafe_allow_html=True)
    
    # Показываем статус загрузки модели
    if NeuralNetwork is not None:
        st.success("Модель успешно загружена!")
    
    st.image(
        'miscellaneous/attention.webp',
        caption='"Саломатӣ беҳтар аз ҳама неъматҳост. —Артур Шопенгауэр"',
        use_column_width=True
    )
    
    st.markdown("<h3 style='text-align: center; color: white;'>Модел чӣ гуна кор мекунад?</h3>", unsafe_allow_html=True)
    st.write("Истифодабаранда бояд расми рентгениро бор кунад ва модел натиҷаи эҳтимолияти бемориро нишон медиҳад.")
    
    # Намоиши мисолҳо
    col1, col2 = st.columns(2)
    with col1:
        try:
            col1.image(resize_image('xray_samples/Healthy.jpg'), caption='Кафаси синаи шахси солим')
        except Exception as e:
            st.error(f"Ошибка загрузки примера изображения: {str(e)}")
    
    with col2:
        try:
            col2.image(resize_image('xray_samples/Pneumonia.jpeg'), caption='Кафаси синаи шахси бемор')
        except Exception as e:
            st.error(f"Ошибка загрузки примера изображения: {str(e)}")
    
    # Боргузорӣ ва пешгӯӣ
    st.markdown("<h5 style='text-align: center; color: white;'>Расмро бор кунед</h5>", unsafe_allow_html=True)
    predict('акс')  # Номи оддии тасвир барои боркунӣ

if __name__ == "__main__":
    main()
