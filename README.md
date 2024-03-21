# 🩺 Pneumonia Detection Computer Vision

In the realm of medical diagnostics, where precision is paramount, I embarked on a journey to develop a Computer Vision model capable of detecting viral or bacterial pneumonia. The project unfolded in several meticulously planned stages:

## 🛠️ Data Engineering
The foundation of any machine learning model lies in its data. Recognizing this, I sourced high-quality X-Ray data from the esteemed Xian Djo University and the diverse Kaggle.com platform. This ensured a robust and comprehensive dataset that would serve as the bedrock for the model.

## 🔍 Exploratory Data Analysis (EDA)
With the data in hand, I delved into an intensive EDA phase. This involved scrutinizing the data, identifying patterns, and understanding the underlying structures. The insights gleaned from this stage were instrumental in shaping the subsequent steps of the project.

## 📊 Data Normalization
To ensure the model wasn't swayed by variations in the scale of features, I implemented data normalization. This process adjusted the values in the dataset to a common scale, without distorting the differences in the range of values or losing information.

## 🔄 Data Augmentation
To further enhance the model's ability to generalize, I employed data augmentation techniques. This involved creating new data instances via transformations such as rotations and translations. This not only expanded the dataset but also enabled the model to learn from a more diverse set of examples.

## 🧠 Model Training using Convolutional Neural Networks (CNN)
Armed with a well-prepared dataset, I initiated the model training phase. I chose Convolutional Neural Networks (CNN), renowned for their efficacy in image analysis tasks. The model was trained to learn from the X-ray images and accurately classify them as indicative of viral or bacterial pneumonia.

This project was a testament to the power of machine learning in healthcare, potentially paving the way for faster, more accurate pneumonia detection. The journey, though challenging, was incredibly rewarding, and the results hold promising implications for the future of medical diagnostics.
