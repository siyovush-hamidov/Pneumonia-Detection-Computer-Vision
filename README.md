# Abstract

![Image](https://www.verywellhealth.com/thmb/NSlwxjm2s163O0MoR-xb7DGMMoc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-1213857828-8c2fa6351d494e5191e1ced8e001dd70.jpg)

This project was developed to participate in the "Student and Technical Progress" competition, aimed at fostering technological advancements. The inspiration for this idea stemmed from the pressing issue of improving medical education in my country, the Republic of Tajikistan. Considering the economic constraints of my nation, the affordability and accessibility of such software solutions are vital. This work represents a step towards leveraging technology to address healthcare challenges, particularly in medical diagnostics.

The project focuses on building a Computer Vision model for pneumonia detection using chest X-ray images. Harnessing the power of Convolutional Neural Networks (CNNs), the model classifies X-ray scans to identify signs of viral or bacterial pneumonia. The model was trained on a robust dataset sourced from Xian Djo University and Kaggle.com, undergoing meticulous preprocessing stages including data normalization and augmentation.

To present the results in an accessible format, a Streamlit-based web application was developed. This interface allows healthcare professionals to upload X-ray images and receive diagnostic insights, making the tool user-friendly and practical for real-world applications.

Technologies such as TensorFlow, Keras, and OpenCV were integral to the project, supporting neural network construction, image preprocessing, and visualization. The model's performance was evaluated using metrics like classification accuracy and confusion matrix analysis, ensuring its reliability in clinical settings.

This initiative underscores the potential of artificial intelligence in revolutionizing healthcare, offering a cost-effective, scalable, and innovative solution to improve diagnostic accuracy and medical education in resource-constrained regions.
# Methodology

## Data Preprocessing
### 1. **Dataset Preparation:**  
- The dataset consisted of grayscale images resized to a uniform size of 150x150 pixels.  
- Each image was normalized to have pixel values between 0 and 1 to ensure consistent input for the model.

```python
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))  # Resizing
    return image / 255.0  # Normalizing
```


### 2. **Augmentation:**  
Image augmentation techniques such as horizontal flipping, rotation, and zoom were applied using the `ImageDataGenerator` from Keras to increase the diversity of the training set and reduce overfitting.

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

### 3. **Splitting:**  
The dataset was split into training, validation, and test sets with proportions of 70%, 20%, and 10%, respectively.
---
## Model Architecture

### **Base Model**  
A Convolutional Neural Network (CNN) was implemented using the Keras Sequential API.  

### **Layers Configuration**  
1. Five convolutional layers with ReLU activation for non-linear feature extraction.  
2. Batch normalization layers to stabilize and accelerate training.  
3. MaxPooling layers to downsample feature maps after each convolutional block.  
4. Dropout layers to mitigate overfitting.  

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
---
## Model Compilation

### **Loss Function**  
Binary Crossentropy was used as the loss function, suitable for binary classification tasks.

### **Optimizer**  
RMSprop was chosen for optimization due to its effectiveness in training deep networks.

```python
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### **Evaluation Metric**  
Accuracy was the primary metric used to evaluate the model's performance.

---

## Training

### **Batch Size and Epochs**  
- Batch size: 32  
- Number of epochs: 12  

### **Learning Rate Scheduler**  
The `ReduceLROnPlateau` callback was implemented to reduce the learning rate when validation accuracy plateaued.  

```python
from keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.3,
    patience=2,
    min_lr=1e-12
)
```

### **Validation**  
Validation was performed after each epoch using the validation dataset to monitor generalization performance.

---

## Evaluation

### **Test Performance**  
The trained model was evaluated on the test set to compute the loss and accuracy.

### **Classification Report**  
Metrics such as precision, recall, and F1-score were calculated for each class to gain deeper insights into the model’s performance.  

```python
# Example: Generating a classification report
from sklearn.metrics import classification_report

predictions = (model.predict(test_images) > 0.5).astype("int32")
print(classification_report(test_labels, predictions, target_names=["Pneumonia", "Normal"]))
```

### **Visualizing Results**  
- Learning curves were plotted to analyze training and validation performance over epochs.  
![Training and Validation Loss](https://i.ibb.co/qncnMz6/image-2024-12-25-15-39-29.png)
![Model Accuracy](https://i.ibb.co/hyMHGgj/image-2024-12-25-15-40-04.png)

- Confusion matrix plotted for a detailed class-wise performance summary.  

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Pneumonia", "Normal"], yticklabels=["Pneumonia", "Normal"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```
![Confusion Matrix](https://i.ibb.co/syyBbk3/image-2024-12-25-15-40-29.png)

# Results  

## 1. Model Performance on Validation Data  

| Metric               | Value       |  
|-----------------------|-------------|  
| **Training Accuracy** | 96.66%      |  
| **Validation Accuracy** | 75.00%     |  
| **Validation Loss**   | 0.4328      |  

---

## 2. Test Dataset Evaluation  

| Metric      | Value       |  
|-------------|-------------|  
| **Test Accuracy** | 90.22% |  
| **Test Loss**     | 0.2470 |  

---

## 3. Classification Report  

| Class                  | Precision | Recall | F1-score |  
|------------------------|-----------|--------|----------|  
| **Pneumonia (Class 0)** | 93%       | 92%    | 92%      |  
| **Normal (Class 1)**    | 86%       | 88%    | 87%      |  

### Overall Metrics  

| Metric            | Accuracy | Macro Average | Weighted Average |  
|--------------------|----------|---------------|------------------|  
| **Overall Metrics** | 90%      | 90%           | 90%              |  

---

## 4. Confusion Matrix  

|                       | Predicted Pneumonia | Predicted Normal |  
|-----------------------|---------------------|------------------|  
| **Actual Pneumonia**  | 358                 | 32               |  
| **Actual Normal**     | 28                  | 206              |  

# Useful Links  

- **Streamlit Application (Ready for Usage):**  
  [Pneumonia Detection App](https://siyovush-hamidov-pneumonia-detection-computer-vision.streamlit.app/)  
