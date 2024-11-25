# CIFAR-10 Classification using Convolutional Neural Networks (CNN)

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset contains 60,000 32x32 color images categorized into 10 classes. The objective is to develop a deep learning model capable of achieving high accuracy in classifying these images.

## Dataset

The CIFAR-10 dataset is composed of 60,000 images distributed across the following 10 classes:
- **Airplane**
- **Automobile**
- **Bird**
- **Cat**
- **Deer**
- **Dog**
- **Frog**
- **Horse**
- **Ship**
- **Truck**

### Key Features:
- **Training Set**: 50,000 images
- **Test Set**: 10,000 images
- **Image Size**: 32x32 pixels
- **Labels**: Pre-labeled for supervised learning tasks

## Model Architecture

The CNN model is structured to classify images accurately into the 10 categories. Key components of the architecture include:
- **Convolutional Layers**: Extract features from input images.
- **Pooling Layers**: Downsample feature maps to reduce spatial dimensions.
- **Fully Connected Layers**: Perform high-level reasoning based on extracted features.
- **Dropout Layers**: Prevent overfitting by randomly disabling neurons during training.
- **Softmax Activation**: Output layer to compute probabilities for each class.

## Results

- **Accuracy Achieved**: The model attained an accuracy of **71.85%** on the test dataset after 4 epochs of training.
- **Optimization Techniques**: Regularization through dropout layers and fine-tuned hyperparameters contributed to achieving this result.


