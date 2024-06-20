
# Capstone Project

This repository contains the code and data for the capstone project, which involves building, training, and deploying a machine learning model using Keras and TensorFlow Lite.

## Project Overview

The goal of this project is to develop a machine learning model that can predict certain outcomes based on the provided dataset. The model is trained using Keras, and its performance is evaluated before converting it into a TensorFlow Lite model for deployment on mobile or embedded devices.

## Repository Contents

- `Modified_DatasetCapstoneRianco.csv`: The dataset used for training and evaluating the machine learning model.
- `capstone.ipynb`: The Jupyter notebook containing the code for data preprocessing, model training, evaluation, and conversion to TensorFlow Lite.

## Installation

To run the code in this repository, you'll need to have Python installed along with the following libraries:

- numpy
- pandas
- tensorflow
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using pip:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn
```

## Usage

1. **Data Preprocessing**: The dataset is loaded and preprocessed to prepare it for training. This includes handling missing values, normalizing data, and splitting it into training and testing sets.

2. **Model Training**: A Keras model is defined, compiled, and trained using the training data. Various metrics are used to evaluate the model's performance.

3. **Model Evaluation**: The trained model is evaluated using the test data to determine its accuracy and other performance metrics.

4. **Model Conversion**: The trained Keras model is converted to a TensorFlow Lite model, making it suitable for deployment on mobile or embedded devices.

## Notebook Walkthrough

### Data Preprocessing

The dataset is first loaded and inspected for any missing values or anomalies. Necessary preprocessing steps are applied to ensure the data is ready for model training.

### Model Training

A neural network model is defined using the Keras Sequential API. The model is compiled with appropriate loss functions and optimizers, and then trained on the preprocessed data.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

### Model Evaluation

The model's performance is evaluated on the test set, and various metrics such as Mean Absolute Error (MAE) are calculated.

```python
mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Mean Absolute Error: {mae}')
```

### Model Conversion

The trained Keras model is converted to a TensorFlow Lite model for deployment.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Results

The results of the model training and evaluation are documented, including any relevant metrics and observations.

## Conclusion

This project demonstrates the process of building, training, and deploying a machine learning model using Keras and TensorFlow Lite. The notebook provides a step-by-step guide to achieving this, from data preprocessing to model conversion.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
