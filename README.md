Capstone Project
This repository contains the code and data for the capstone project, which involves building, training, and deploying a machine learning model using Keras and TensorFlow Lite.

Project Overview
The goal of this project is to develop a machine learning model that can predict certain outcomes based on the provided dataset. The model is trained using Keras, and its performance is evaluated before converting it into a TensorFlow Lite model for deployment on mobile or embedded devices.

Repository Contents
Modified_DatasetCapstoneRianco.csv: The dataset used for training and evaluating the machine learning model.
capstone.ipynb: The Jupyter notebook containing the code for data preprocessing, model training, evaluation, and conversion to TensorFlow Lite.
Installation
To run the code in this repository, you'll need to have Python installed along with the following libraries:

numpy
pandas
tensorflow
scikit-learn
matplotlib
seaborn
You can install these dependencies using pip:

bash
Salin kode
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
Usage
Data Preprocessing: The dataset is loaded and preprocessed to prepare it for training. This includes handling missing values, normalizing data, and splitting it into training and testing sets.

Model Training: A Keras model is defined, compiled, and trained using the training data. Various metrics are used to evaluate the model's performance.

Model Evaluation: The trained model is evaluated using the test data to determine its accuracy and other performance metrics.

Model Conversion: The trained Keras model is converted to a TensorFlow Lite model, making it suitable for deployment on mobile or embedded devices.
