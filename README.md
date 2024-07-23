# FSL-for-HAR-with-DP
#Federated Split Learning for Human Activity Recognition with Differential Privacy

This repository contains an implementation of a federated split learning model with optional differential privacy. The model is designed to work with the Human Activity Recognition (HAR) dataset from the UCI Machine Learning Repository. The code includes functionalities for downloading and preprocessing the dataset, defining LSTM-based client and server models, training the model with federated learning, and optionally applying differential privacy techniques.

#Dataset

The dataset used in this project is the Human Activity Recognition Using Smartphones Data Set. It contains sensor data collected from the accelerometers and gyroscopes of smartphones worn by 30 participants while performing six different activities.

#Requirements

Python 3.7+
PyTorch 1.8.0+

#Key Features

Federated split learning setup with client-server model architecture.
Optional differential privacy with adjustable parameters for epsilon, delta, and gradient clipping norm.
Functions for downloading and extracting the HAR dataset, loading the data, and splitting it among clients.
Training loop with federated aggregation of model weights and performance evaluation.

#Usage

To start the training process, download and extract the dataset, load the data, and train the model with federated learning:
Uncomment the start() function call in the main script to run the training process.

This README provides a brief overview of the project, the dataset, key features, and usage instructions. Feel free to adjust the content as needed for your specific requirements.
