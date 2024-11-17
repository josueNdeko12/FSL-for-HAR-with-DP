# 💫 About Me:
Former AI Research Intern | Honors CS | Dell Scholar | CDEP Scholar at FVSU | NSBE


# 💻 Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)
# 📊 GitHub Stats:
![](https://github-readme-stats.vercel.app/api?username=josuendeko12&theme=dark&hide_border=true&include_all_commits=false&count_private=false)<br/>
![](https://github-readme-streak-stats.herokuapp.com/?user=josuendeko12&theme=dark&hide_border=true)<br/>
![](https://github-readme-stats.vercel.app/api/top-langs/?username=josuendeko12&theme=dark&hide_border=true&include_all_commits=false&count_private=false&layout=compact)

## 🏆 GitHub Trophies
![](https://github-profile-trophy.vercel.app/?username=josuendeko12&theme=radical&no-frame=false&no-bg=false&margin-w=4)

### ✍️ Random Dev Quote
![](https://quotes-github-readme.vercel.app/api?type=horizontal&theme=dark)

<!-- Proudly created with GPRM ( https://gprm.itsvg.in ) -->

# FSL-for-HAR-with-DP
# Federated Split Learning for Human Activity Recognition with Differential Privacy

This repository contains an implementation of a federated split learning model with optional differential privacy. The model is designed to work with the Human Activity Recognition (HAR) dataset from the UCI Machine Learning Repository. The code includes functionalities for downloading and preprocessing the dataset, defining LSTM-based client and server models, training the model with federated learning, and optionally applying differential privacy techniques.

# Dataset

The dataset used in this project is the Human Activity Recognition Using Smartphones Data Set. It contains sensor data collected from the accelerometers and gyroscopes of smartphones worn by 30 participants while performing six different activities.

Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

# Requirements

Python 3.7+
PyTorch 1.8.0+

# Key Features

Federated split learning setup with client-server model architecture.
Optional differential privacy with adjustable parameters for epsilon, delta, and gradient clipping norm.
Functions for downloading and extracting the HAR dataset, loading the data, and splitting it among clients.
Training loop with federated aggregation of model weights and performance evaluation.

# Usage

To start the training process, download and extract the dataset, load the data, and train the model with federated learning:
Run the main.py script to start the training process.

This README provides a brief overview of the project, the dataset, key features, and usage instructions. Feel free to adjust the content as needed for your specific requirements.
