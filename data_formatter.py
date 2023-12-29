from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import librosa


NUM_TRAINING = 2560
NUM_VALIDATION = 1168
NUM_TESTING = 16
svm_array_length = 3


#  Parse the training .wav file names and their corresponding labels from train.txt
def get_train_labels():
    fd = open('train.txt')
    file_names = []
    labels = []
    for line in fd.readlines():
        info = line.strip('\n').split(',')
        file_names.append(info[0])
        labels.append(info[1])
    fd.close()
    return file_names, labels


#  Parse the training .wav file names and their corresponding labels from validation.txt
def get_validation_labels():
    fd = open('validation.txt')
    file_names = []
    labels = []
    for line in fd.readlines():
        info = line.strip('\n').split(',')
        file_names.append(info[0])
        labels.append(info[1])
    fd.close()
    return file_names, labels


#   Get the testing file names
def get_testing_files():
    fd = open('test.txt')
    file_names = []
    for line in fd.readlines():
        info = line.strip('\n')
        file_names.append(info)
    fd.close()
    return file_names


#  Function that prints the predicted labels into a submittable csv file
def write_predicted_labels(testing_file_names, labels, file_name):
    fd = open(file_name, 'w')
    fd.write('name,label\n')
    for i in range(NUM_TESTING):
        fd.write(testing_file_names[i] + ',' + str(labels[i]) + '\n')
    fd.close()


#  The following function is inspired from: https://stackoverflow.com/questions/50355543/binary-classification-of-audio-wav-files

#  With the default width and height values, it generates a 150x150 melspectrogram.
def get_melspectrogram(path, fixed_width = 150, fixed_height = 150):
    #  Our images' sample rate is 16000, which is different to the library's default
    signal, sample_rate = librosa.load(path, sr = 16000)
    hop_length = int(signal.shape[0] / (fixed_width * 1.1))
    spectrogram = librosa.feature.melspectrogram(y=signal, n_mels = fixed_height, hop_length = hop_length)
    spectrogram = librosa.power_to_db(spectrogram)

    return spectrogram[:, :150]


#  This functions generates and saves the melspectrograms used by the neural network.
def audio_to_images_librosa():
    training_file_names, training_labels = get_train_labels()
    validation_file_names, validation_labels = get_validation_labels()
    testing_file_names = get_testing_files()

    print('Generating melspectrograms for training...')
    training_images = np.zeros((NUM_TRAINING, 150, 150))
    for i, file_name in enumerate(training_file_names):
        training_images[i] = get_melspectrogram(file_name)


    print('Generating melspectrograms for validation...')
    validation_images = np.zeros((NUM_VALIDATION, 150, 150))
    for i, file_name in enumerate(validation_file_names):
        validation_images[i] = get_melspectrogram(file_name)


    print('Generating melspectrograms for testing...')
    testing_images = np.zeros((NUM_TESTING, 150, 150))
    for i, file_name in enumerate(testing_file_names):
        testing_images[i] = get_melspectrogram(file_name)


    print('Saving melspectrograms...')
    #  Save images
    np.save('train_mels_150/train.npy', training_images)
    np.save('validation_mels_150/validation.npy', validation_images)
    np.save('test_mels_150/test.npy', testing_images)

    print('Saving labels...')
    #  Save labels
    np.save('train_mels_150/train_labels.npy', np.asarray(training_labels, dtype = 'int'))
    np.save('validation_mels_150/validation_labels.npy', np.asarray(validation_labels, dtype = 'int'))


if __name__ == '__main__':
    audio_to_images_librosa()