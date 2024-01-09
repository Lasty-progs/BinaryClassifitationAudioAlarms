import numpy as np
import librosa, os
import pandas as pd


CLASSIFICATION_CLASS = "Alarm" # Chose one class from vocabulary.csv
NUM_TESTING = 16
SIZE_IMAGES = (300,300)


def get_file_names() -> dict:
    '''Get files from dataset folder(./FSD50K/dev_audio/ and ./FSD50K/eval_audio/)'''
    files = {"dev":os.listdir("./FSD50K/" + "dev" + "_audio/"),
             "eval":os.listdir("./FSD50K/" + "eval" + "_audio/")}
    return files

def count_true_examples(df, classification_class) -> (int, int):
    '''Count positive labels for balance dataset'''
    train = df[df.split == 'train']
    val = df[df.split == 'val']
    train_count = val_count = 0
    for _, row in train.iterrows():
        train_count += 1 if classification_class in row['labels'].split(',') else 0
    for _, row in val.iterrows():
        val_count += 1 if classification_class in row['labels'].split(',') else 0
    return train_count, val_count


def get_dev_labels(classification_class:str):
    '''Classification class chose from vocabulary.csv'''

    files = get_file_names()['dev']
    df = pd.read_csv("./FSD50K/ground_truth/dev.csv")

    train_count, val_count = count_true_examples(df, classification_class)

    train_info = {"names" : [], "labels" : []}
    validation_info =  {"names" : [], "labels" : []}

    for _, row in df.iterrows():
        fname = str(row['fname'])+".wav"
        if (fname) in files:
            if row['split'] == "train" and train_count > 0:
                train_info['names'].append("FSD50K/dev_audio/" + fname)
                categories = row['labels'].split(',')
                train_count -= 0 if classification_class in categories else 1
                train_info['labels'].append(classification_class in categories)
            elif row['split'] == "val" and val_count > 0:
                validation_info['names'].append("FSD50K/dev_audio/" + fname)
                categories = row['labels'].split(',')
                val_count -= 0 if classification_class in categories else 1
                validation_info['labels'].append(classification_class in categories)

    return (train_info['names'], train_info['labels'],
            validation_info['names'], validation_info['labels'])


def get_test_files():

    files = get_file_names()['eval']
    df = pd.read_csv("./FSD50K/ground_truth/eval.csv")
    test_files = []

    for _, row in df.iterrows():
        fname = str(row['fname'])+".wav"
        if (fname) in files:
            test_files.append("FSD50K/eval_audio/" + fname)

    return test_files

def generate_folders():
    try:
        os.makedirs("./temp")
        os.makedirs("./temp/train_mels_150")
        os.makedirs("./temp/validation_mels_150")
        os.makedirs("./temp/test_mels_150")
    except:
        print("User warning: Error in creation folders(they are already exists)")


#  The following function is inspired from: https://stackoverflow.com/questions/50355543/binary-classification-of-audio-wav-files

#  With the default width and height values, it generates a 150x150 melspectrogram.
def get_melspectrogram(path, fixed_width = 150, fixed_height = 150):
    signal, sample_rate = librosa.load(path, sr = 16000)
    hop_length = int(signal.shape[0] / (fixed_width * 1.1))
    spectrogram = librosa.feature.melspectrogram(y=signal, n_mels = fixed_height, hop_length = hop_length)
    spectrogram = librosa.power_to_db(spectrogram)

    return spectrogram[:, :fixed_width]


#  This functions generates and saves the melspectrograms used by the neural network.
def audio_to_images_librosa(size:tuple):
    (training_file_names, training_labels, validation_file_names,
     validation_labels) = get_dev_labels(CLASSIFICATION_CLASS)
    testing_file_names = get_test_files()
    generate_folders()

    # num_training, num_validation = count_true_examples(pd.read_csv("./FSD50K/ground_truth/dev.csv"),
    #                                                    CLASSIFICATION_CLASS)

    print('Generating melspectrograms for training...')
    training_images = np.zeros((len(training_labels), size[0], size[1]))
    for i, file_name in enumerate(training_file_names):
        training_images[i] = get_melspectrogram(file_name, fixed_height=size[0], fixed_width=size[1])


    print('Generating melspectrograms for validation...')
    validation_images = np.zeros((len(validation_labels), size[0], size[1]))
    for i, file_name in enumerate(validation_file_names):
        validation_images[i] = get_melspectrogram(file_name, fixed_height=size[0], fixed_width=size[1])


    print('Generating melspectrograms for testing...')
    testing_images = np.zeros((len(testing_file_names), size[0], size[1]))
    for i, file_name in enumerate(testing_file_names):
        testing_images[i] = get_melspectrogram(file_name, fixed_height=size[0], fixed_width=size[1])


    print('Saving melspectrograms...')
    #  Save images
    np.save('temp/train_mels_150/train.npy', training_images)
    np.save('temp/validation_mels_150/validation.npy', validation_images)
    np.save('temp/test_mels_150/test.npy', testing_images)

    print('Saving labels...')
    #  Save labels
    np.save('temp/train_mels_150/train_labels.npy', np.asarray(training_labels, dtype = 'int'))
    np.save('temp/validation_mels_150/validation_labels.npy', np.asarray(validation_labels, dtype = 'int'))


if __name__ == '__main__':
    audio_to_images_librosa(SIZE_IMAGES)
