import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import librosa
from scipy.signal import spectrogram



train_img = np.load('train_mels_150/train.npy')[123]

# print(train_img.shape)
mini = np.min(train_img)

# for i in train_img:
#     print(i)





signal, sample_rate = librosa.load("fsd50K/audio/test/125520.wav", sr=16000)
hop_length = int(signal.shape[0] / (150 * 1.1))
spectrogram = librosa.feature.melspectrogram(y=signal,n_mels = 150, hop_length = hop_length)
spectrogram = librosa.power_to_db(spectrogram)
# spectr += 50
# print(spectr.shape)


plt.imsave("test_mell.jpg", spectrogram, cmap='viridis')