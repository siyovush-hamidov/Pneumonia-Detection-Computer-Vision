import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
import os

# Загрузка модели
NeuralNetwork = load_model('PneuClass(90%).h5')
img_size = 150

# Проверяем наличие файлов
assert os.path.exists('Healthy.jpg'), "Файл Healthy.jpg не найден."
assert os.path.exists('Pneumonia.jpeg'), "Файл Pneumonia.jpeg не найден."

def resize_image(img_path, max_width=400):
    """Уменьшает изображение для отображения."""
    img = Image.open(img_path)
    if img.width > max_width:
        scale = max_width / img.width
        new_size = (max_width, int(img.height * scale))
        img = img.resize(new_size, Image.ANTIALIAS)
    return img

def predict(name):
    """Функция для загрузки и предсказания по изображению."""
    image = st.file_uploader("Загрузите фотографию " + name, type=["png", "jpg", "jpeg"])
    if image:
        st.image(image=image)
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
                st.markdown("<h4 style='text-align: center; color: white;'>Обнаружены признаки пневмонии.</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: white;'>Вероятность наличия составляет {ANS}%</h5>", unsafe_allow_html=True)
            else:
                st.markdown("<h4 style='text-align: center; color: white;'>Признаков заболевания не обнаружено.</h4>", unsafe_allow_html=True)
                st.markdown(f"<h5 style='text-align: center; color: white;'>Вероятность наличия составляет {ANS}%</h5>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ошибка при работе с моделью: {str(e)}")

def main():
    """Главная функция для отображения страницы Streamlit."""
    st.markdown("<h2 style='text-align: center; color: white;'>Модель машинного обучения для диагностирования бактериальной и вирусной пневмонии</h2>", unsafe_allow_html=True)
    st.image(
        'attention.webp',
        caption='“Здоровье перевешивает все остальные блага жизни. —Артур Шопенгауэр”',
        use_column_width=True
    )
    st.markdown("<h3 style='text-align: center; color: white;'>Как работает модель?</h3>", unsafe_allow_html=True)
    st.write("Пользователь должен загрузить рентгеновский снимок, чтобы получить результат в виде вероятности наличия заболевания.")
    
    # Отображение изображений для примера
    col1, col2 = st.columns(2)
    col1.image(resize_image('Healthy.jpg'), caption='Снимок грудной клетки здорового пациента')
    col2.image(resize_image('Pneumonia.jpeg'), caption='Снимок грудной клетки, пораженной болезнью')

    # Загрузка и предсказание
    st.markdown("<h5 style='text-align: center; color: white;'>Загрузите снимок</h5>", unsafe_allow_html=True)
    predict('image')  # Передаем имя загруженного файла

if __name__ == "__main__":
    main()
