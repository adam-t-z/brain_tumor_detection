# 🧠 Brain Tumor Classification using MRI Images

Welcome to the **Brain Tumor Classification** project! This project uses machine learning to classify brain MRI images into two categories: **with tumor** and **without tumor**. The goal is to predict whether a given MRI scan shows signs of a brain tumor using **Logistic Regression**.

## 📂 Dataset

The dataset used in this project is from Kaggle and can be accessed using the following link:

[Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

It contains a set of MRI images of the brain, which are categorized into different types of tumors and "no tumor" classes.

## ⚙️ How It Works

The process follows these steps:

1. **Data Import**: We load the necessary libraries and the dataset.
2. **Data Preprocessing**: Images are read, resized to a standard size (128x128), and converted to grayscale. The pixel values are normalized for better model performance.
3. **Train/Test Split**: 80% of the data is used for training, and 20% is reserved for testing the model.
4. **Model Training**: Logistic Regression is used to classify the images based on the training data.
5. **Evaluation**: The model's performance is evaluated on both the training and testing datasets.

## 📊 Results

After training the model, the **Logistic Regression** model achieved an impressive **94% accuracy** on the testing dataset!

## 💾 Model Saving

Once the model is trained, it is saved as `tumor_classifier.pkl` using **joblib**, so you can easily load and use it for predictions without retraining.

## 🛠️ Requirements

To run this project, you'll need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python (cv2)`
- `scikit-learn`
- `joblib`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn joblib
```
## 🚀 Possible Future Improvements

- **Try using other machine learning models**: Explore different algorithms like **SVM** (Support Vector Machine) and **Random Forest** and compare their performance to see if they yield better results.

- **Implement image augmentation techniques**: Apply image transformations (like rotation, flipping, zooming) to increase the diversity of training data and help improve the model’s ability to generalize to new images.

- **Explore deep learning models**: Dive into **Convolutional Neural Networks (CNNs)**, which are specifically designed for image classification tasks and might provide even better accuracy than traditional models like Logistic Regression.
