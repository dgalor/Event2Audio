#%%
from metavision_sdk_cv import SparseOpticalFlowAlgorithm
import numpy as np
events = np.load("abe_speech.npy")

s, e, f = 3, 5.0, 1e5 #12.0

cond = (events["t"]>=s*1e6) & (events["t"]<e*1e6)
events = events[cond]
events["t"] = events["t"] - events["t"].min()

e, s = events["t"].max()/1e6, events["t"].min()/1e6
nbins = int(round((e-s)*f))
relative_t = (events["t"]/1e6-s)*f
coord_t = np.round(relative_t).astype(np.uint32)
events["x"] = events["x"]-events["x"].min()
events["y"] = events["y"]-events["y"].min()
w, h = events["x"].max()+1, events["y"].max()+1
# %%
from tqdm import trange
from metavision_sdk_cv import SparseOpticalFlowAlgorithmParameters
sparse_flow = SparseOpticalFlowAlgorithm(w, h, tuning = SparseOpticalFlowAlgorithmParameters.Tuning.SlowObjects)
# sparse_flow = SparseOpticalFlowAlgorithm(
#     w, h,
#     distance_gain = 0.05000000074505806,
#     damping = 0.7070000171661377,
#     omega_cutoff = 7.0*100,
#     min_cluster_size = 7,
#     max_link_time = 30000,
#     match_polarity = True,
#     use_simple_match = True,
#     full_square = True,
#     last_event_only = False,
#     size_threshold = 100000000
# )
batch_duration = 0.5e6 # 300ms
nbatches = np.ceil(events["t"].max()/batch_duration).astype(int)
out = []
for i in trange(nbatches):
    batch = events[(events["t"]>=i*batch_duration) & (events["t"]<(i+1)*batch_duration)]
    buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()
    sparse_flow.process_events(batch, buffer)
    out.append(buffer.numpy().copy())
out = np.concatenate(out)
# %%
%%time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
tv, vx = out["t"], out["vx"]
coord_tv = np.round((tv/1e6-s)*f).astype(np.uint32)
counts = coo_matrix((np.ones_like(coord_tv), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat = coo_matrix((vx, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat_amp = coo_matrix((np.abs(vx), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
mat = mat_amp / counts.clip(1, None) * np.sign(mat)
# mat = mat / counts.clip(1, None)
plt.plot(mat)
# %%
%%time
import matplotlib.pyplot as plt
import scipy.signal as signal
pt = mat
pt = pt / np.abs(pt).max()
n_fft = 2080  # FFT window size
hop_length = 112  # Number of samples between successive frames
a = pt
# Calculate spectrogram
f_, t_, Sxx = signal.spectrogram(a,
                                    fs=f,
                                    nperseg=n_fft,
                                    noverlap=n_fft-hop_length,
                                    window="hann")
# Plot
plt.pcolormesh(t_, f_, 10 * np.log10(Sxx), cmap="magma", shading="auto")
plt.colorbar(label="Intensity [dB]")
plt.ylim([0, 6000])  # Match frequency range with mel spectrogram
plt.title("Recovered Chirp (0-20k)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
#%%
#now save audio in wav
import soundfile as sf
sf.write("abe_speech.wav", a, int(f))
#%%
# import librosa
# n_mels = 100
# fmax = 4096
# fmin = 100
# sr = f
# y = mat
# y = y / np.abs(y).max()
# fig, ax = plt.subplots(figsize=(12,5))
# M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin, n_fft=8048)
# M_db = librosa.power_to_db(M, ref=np.max)
# img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, fmax=fmax,fmin=fmin,sr=sr)
# plt.colorbar(img,ax=ax)
# plt.title('Speech Recovery (Ours)')
# plt.show()
# #%%
# flow_np = out
# xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
# coords = np.stack((ys, xs))
# abs_coords = np.ravel_multi_index(coords, (h, w))
# counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
#                         minlength=h*w).reshape(h, w)
# flow_x = np.bincount(abs_coords, weights=vx, minlength=h*w).reshape(h, w)
# flow_y = np.bincount(abs_coords, weights=vy, minlength=h*w).reshape(h, w)
# mask_multiple_events = counts > 1
# flow_x[mask_multiple_events] /= counts[mask_multiple_events]
# flow_y[mask_multiple_events] /= counts[mask_multiple_events]

# flow = np.stack((flow_x, flow_y)).astype(np.float32)# %%

# # %%
