<img align="center" width="50%" src='Images/jan-huber-SqR_XkrwwPk-unsplash.jpg' alt="title" />

# Audio Classification using Wavelet Transform and Deep Learning
A step-by-step tutorial to classify audio signals using continuous wavelet transform (CWT) as features.

<hr>

- ## Steps to use this repository:

    - Create a virtual environment by using the command: ```virtualenv venv```
    - Activate the environment: ```source venv/bin/activate```
    - Install the requirements.txt file by typing: ```pip install -r requirements.txt```
    - Extract the recordings.zip file

- ## Files Description

    - recordings.zip: The contains recordings from the Free Spoken Digit Dataset (FSDD). You can also find this data [here](https://github.com/Jakobovski/free-spoken-digit-dataset). 
    - training_raw_audio.npz: We are only classifying 3 speakers here: george, jackson, and lucas. All the training data from these 3 speakers is in this numpy zip file.
    - testing_raw_audio.npz: We are only classifying 3 speakers here: george, jackson, and lucas. All the testing data from these 3 speakers is in this numpy zip file.
    - requirements.txt: It contains the required libraries.
