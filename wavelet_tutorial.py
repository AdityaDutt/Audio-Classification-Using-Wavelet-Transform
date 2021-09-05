
# Import libraries
import os, sys, cv2, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sn
from pandas.core import frame
from librosa.core import audio
import random
from random import seed, random, randint, sample
from scipy.spatial import distance

import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Multiply, Add, dot, GlobalMaxPool1D, Dropout, Masking, Activation, MaxPool1D, Conv1D, Flatten, TimeDistributed, Lambda
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

import librosa 
import librosa.display
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import resize
from scipy.signal import hilbert, chirp
from sklearn.preprocessing import MinMaxScaler
from librosa.filters import mel
import pywt
import scipy
from tqdm import tqdm

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift



'''
# The data in the current directory inside the doler "recordings".
dir = os.getcwd() + "/recordings/"

# Read audio files from the directory. For this tutorial, we will only classify 3 speakers: george, jackson, and lucas.
# Audio files have this format : {digit}_{speaker}_{speaker_filenumber}.wav

audio = [] # List to store audio np arrays
y = [] # List to store the target class labels

for root, dirs, files in os.walk(dir, topdown=False):    
    for name in files:

        if name.find(".wav") != -1 : # Check if the file has a .wav extension            
            if name.find("george") != -1 or name.find("jackson") != -1 or name.find("lucas") != -1 : # Check if the speaker is george, jackson, and lucas.
                fullname = os.path.join(root, name)
                audio.append(fullname) # Append the np array to the list.
                if name.find("george") != -1 :
                    y.append(0)
                elif name.find("jackson") != -1 :
                    y.append(1)
                else :
                    y.append(2)

# Write the audio data in a npz file so that we don't have to read the audio files again. We can load the data from npz file. Also, the npz format is very space efficient.
audio_train, audio_test, y_train, y_test = train_test_split(audio, y, test_size=0.3)
np.savez_compressed(os.getcwd()+"/training_raw_audio", a=audio_train, b=y_train)
np.savez_compressed(os.getcwd()+"/testing_raw_audio", a=audio_test, b=y_train)

print("Finished writing to npz file...")

# Print the class distribution
print("Training Data class distribution: ", np.unique(y_train, return_counts=True))
print("Testing Data class distribution: ", np.unique(y_test, return_counts=True))
'''


# Load the data from the .npz file
train_data = np.load(os.getcwd()+"/training_raw_audio.npz", allow_pickle=True)
audio_train = train_data['a']
y_train = train_data['b']

test_data = np.load(os.getcwd()+"/testing_raw_audio.npz", allow_pickle=True)
audio_test = test_data['a']
y_test = test_data['b']

def compute_wavelet_features(X) :
    
    # Define a few parameters
    wavelet = 'morl' # wavelet type: morlet
    sr = 8000 # sampling frequency: 8KHz
    widths = np.arange(1, 512) # scales for morlet wavelet 
    dt = 1/sr # timestep difference

    frequencies = pywt.scale2frequency(wavelet, widths) / dt # Get frequencies corresponding to scales
    
    # Create a filter to select frequencies between 80Hz and 5KHz
    upper = ([x for x in range(len(widths)) if frequencies[x] > 5000])[-1]
    lower = ([x for x in range(len(widths)) if frequencies[x] < 80])[0]
    widths = widths[upper:lower] # Select scales in this frequency range

    # Compute continuous wavelet transform of the audio numpy array
    wavelet_coeffs, freqs = pywt.cwt(X, widths, wavelet = wavelet, sampling_period=dt)
    wavelet_coeffs = wavelet_coeffs.astype(np.float16)

    # Split the coefficients into frames of length 800
    start = 0
    end = wavelet_coeffs.shape[1]
    frames = []
    frame_size = 800
    count = 0

    while start+frame_size <= end-1 :

        f = wavelet_coeffs[:,start:start+frame_size]
        frames.append(f)

        start += frame_size


    # Convert frames to numpy array
    frames = np.array(frames)
    frames = frames.reshape((len(frames), wavelet_coeffs.shape[0], frame_size))

    return frames



### Training data features

indices = []

for i in range(3) :
    
    ind, = np.where(y_train == i)
    seed(i)
    ind = ind.tolist()
    ind = sample(ind, 100)
    audio_samples = audio_train[ind]
    num_rand_samp = 30
    WaveletFeatTrain = [] # Store wavelet features
    WaveletYTrain = [] # Store class labels corresponding to wavelet features from an audio sample

    for j in tqdm(range(len(audio_samples))) :

        # print("i ", i, " j ", j, "/", len(audio_samples))
        curr_sample = audio_samples[j]
        seq, _ = librosa.load(curr_sample) 
        F = compute_wavelet_features(seq)
        F = F.astype(np.float16)

        # Generate target labels corresponding to the frames of each sample
        indices = np.arange(0, len(F), 1)
        indices = indices.tolist()
        indices = sample(indices, min(num_rand_samp, len(indices)))
        F = F[indices]

        if j == 0 :
            WaveletFeatTrain = F
        else :
            WaveletFeatTrain = np.concatenate((WaveletFeatTrain, F), axis=0) 
        # print(i, "/", len(audio_train), WaveletFeatTrain.shape)
    WaveletYTrain += [i] * len(ind)
    WaveletYTrain = np.array(WaveletYTrain) # Convert to numpy array
    
    print("X: ", WaveletFeatTrain.shape, "  y: ", WaveletYTrain.shape)

    # np.savez_compressed("/Volumes/Aditya/WaveletTutorialData/training_features_class"+str(i), a=WaveletFeatTrain, b=WaveletYTrain)


# Write all features to a .npz file
sys.exit(1)

### Testing data features

WaveletFeatTest = [] # Store wavelet features
WaveletYTest = [] # Store class labels corresponding to wavelet features from an audio sample

for i in range(len(audio_test)) :

    curr_sample = audio_test[i]
    seq, _ = librosa.load(curr_sample) 
    curr_target = y_test[i]
    F = compute_wavelet_features(seq)

    # Generate target labels corresponding to the frames of each sample
    WaveletYTest += [curr_target] * len(F)

    if i == 0 :
        WaveletFeatTest = F
    else :
        WaveletFeatTest = np.concatenate((WaveletFeatTest, F), axis=0) 

WaveletYTest = np.array(WaveletYTest) # Convert to numpy array
print("X: ", WaveletFeatTest.shape, "  y: ", WaveletYTest.shape)

WaveletFeatTest = WaveletFeatTest.astype(np.float16)

# Write all features to a .npz file
np.savez_compressed(os.getcwd()+"/testing_features", a=WaveletFeatTest, b=WaveletYTest)
