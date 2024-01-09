import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import librosa
from scipy.signal import spectrogram



train_img = np.load('temp/validation_mels_150/validation.npy')[2]

# print(train_img.shape)

plt.imsave("test_mell.jpg", train_img, cmap='viridis')




# mini = np.min(train_img)

# for i in train_img:
#     print(i)





# signal, sample_rate = librosa.load("FSD50K/dev_audio/236.wav", sr=16000)
# spectrogram = librosa.feature.melspectrogram(y=signal,n_mels = 500, hop_length = 500)
# spectrogram = librosa.power_to_db(spectrogram)
# print(spectrogram.size)


# plt.imsave("test_mell.jpg", spectrogram, cmap='viridis')