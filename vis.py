#%%
import librosa
import matplotlib.pyplot as plt
import numpy as np
data, f = librosa.load("abe_speech_bag_hat_denoised.wav", sr=None)

n_mels = 100
fmax = 4096
fmin = 100
sr = f
fig, ax = plt.subplots(figsize=(12,5))
M = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin, n_fft=8048)
M_db = librosa.power_to_db(M, ref=np.max)
img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, fmax=fmax,fmin=fmin,sr=sr)
plt.colorbar(img,ax=ax)
plt.title('Speech Recovery (Ours)')
plt.show()
# %%
