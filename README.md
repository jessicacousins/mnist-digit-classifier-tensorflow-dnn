# MNIST Digit Classifier Using TensorFlow.js - Deep Neural Network

## Description

This project demonstrates a deep neural network (DNN) using TensorFlow.js to classify handwritten digits from the MNIST dataset, allowing for real-time classification.

## Features

- Deep Neural Network: Leverages TensorFlow.js for building and training a DNN model.
- MNIST Dataset: Utilizes the popular MNIST dataset, which contains 28x28 pixel greyscale images of handwritten digits (0-9).
- Real-time Classification: Performs real-time digit classification in the browser.
- Visual Feedback: Provides visual feedback by displaying the input image and the prediction results with color-coded accuracy.

## DNN Model Structure

- Input Layer: Takes in 784 features (flattened 28x28 pixel images).
- Hidden Layers:
  - First layer with 32 neurons, ReLU activation.
  - Second layer with 16 neurons, ReLU activation.
- Output Layer: 10 neurons (one for each digit), softmax activation.

## Training Parameters

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Validation Split: 20%
- Batch Size: 512
- Epochs: 50

## Acknowledgments

This project was copied from the Google Developers and TensorFlow.js tutorial, providing me a practical understanding of machine learning concepts and TensorFlow.js applications.

## License

This project is licensed under the Apache License 2.0
