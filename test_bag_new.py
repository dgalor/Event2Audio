#%%
import numpy as np
from metavision_sdk_cv import TripletMatchingFlowAlgorithm, SparseOpticalFlowAlgorithm, TimeGradientFlowAlgorithm, PlaneFittingFlowAlgorithm
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import librosa
import time

sr = 16000
gt = librosa.load("abe_speech_gt.mp3", sr=sr)[0]

events = np.load("abe_speech_bag.npy")

s, e, f = 2.5, 11.7, 1e5 #12.0

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
#%%
def align(x_hat, x):
    from scipy.signal import correlate
    peak_lag = np.inf
    while peak_lag != 0:
        correlation = correlate(x_hat, x, mode='full')
        lags = np.arange(-len(x_hat) + 1, len(x))
        #abs peak
        peak_ind = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_ind]
        if peak_lag < 0:
            x_hat = np.pad(x_hat, (-peak_lag, 0), constant_values=0)#out_der_sr[0])
            x_hat = x_hat[:len(x)]
        else:
            x_hat = np.pad(x_hat, (0, peak_lag), constant_values=0)#out_der_sr[-1])
            x_hat = x_hat[-len(x):]
        if correlation[peak_ind] < 0:
            print("Negative correlation, flipping sign")
            x_hat = -x_hat
        print("Peak lag:", peak_lag)
    x_hat = np.pad(x_hat, (0, len(x)-len(x_hat)), constant_values=0)#out_der_sr[-1])
    return x_hat
def compute_flow(alg, batch_duration=1.0e6):
    t1 = time.time()
    nbatches = np.ceil(events["t"].max()/batch_duration).astype(int)
    out = []
    for i in trange(nbatches):
        batch = events[(events["t"]>=i*batch_duration) & (events["t"]<(i+1)*batch_duration)]
        buffer = alg.get_empty_output_buffer()
        alg.process_events(batch, buffer)
        out.append(buffer.numpy().copy())
    print("Time taken:", time.time()-t1)
    out = np.concatenate(out)
    return out
def cf():
    alg = TimeGradientFlowAlgorithm(w, h, radius = 7, min_flow_mag = 10.0, bit_cut = 0)
    t1 = time.time()
    buffer = alg.get_empty_output_buffer()
    alg.process_events(events, buffer)
    print("Time taken:", time.time()-t1)
    return buffer.numpy().copy()
def mel(out, f=f):
    n_mels = 100
    fmax = 4096
    fmin = 100
    sr = f
    y = out
    y = y / np.abs(y).max()
    fig, ax = plt.subplots(figsize=(12,5))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin, n_fft=8048)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, fmax=fmax,fmin=fmin,sr=sr)
    plt.colorbar(img,ax=ax)
    plt.title('Speech Recovery (Ours)')
    plt.show()
def process(out):
    tv, vx, vy = out["t"], out["vx"], out["vy"]
    coord_tv = np.round((tv/1e6-s)*f).astype(np.uint32)
    counts = coo_matrix((np.ones_like(coord_tv), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
    mat_x = coo_matrix((vx, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
    mat_y = coo_matrix((vy, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
    # mat_amp_x = coo_matrix((np.abs(vx), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
    # mat_amp_y = coo_matrix((np.abs(vy), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
    mat_x = mat_x / counts.clip(1, None)
    mn, std = np.abs(mat_x).mean(), np.abs(mat_x).std()
    mat_x = np.clip(np.abs(mat_x), mn-5*std, mn+5*std) * np.sign(mat_x)
    mat_y = mat_y / counts.clip(1, None)
    mn, std = np.abs(mat_y).mean(), np.abs(mat_y).std()
    mat_y = np.clip(np.abs(mat_y), mn-5*std, mn+5*std) * np.sign(mat_y)
    mat_x = align(mat_x, mat_y)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    mat = pca.fit_transform(np.stack((mat_x, mat_y), axis=-1))[:,0]
    # mag = np.sqrt(mat_x**2 + mat_y**2)
    # mat = mag * (np.sign(mat_y) if np.abs(mat_x).mean() < np.abs(mat_y).mean() else np.sign(mat_x))
    
    # mat_x = mat_x / counts.clip(1, None)
    # mat_x = mat_x #/ np.abs(mat_x).max()
    # mat_y = mat_y / counts.clip(1, None)
    # mat_y = mat_y #/ np.abs(mat_y).max()
    
    # mat_xy = np.stack((mat_x, mat_y), axis=-1)
    # ft = np.fft.fft(mat_xy, axis=0)
    # zm_1, zm_2 = np.abs(ft).T
    # phi_1, phi_2 = np.angle(ft).T
    # theta = np.arctan2(2*zm_1*zm_2*np.cos(phi_1-phi_2), (zm_2**2-zm_1**2))/2
    # assert (np.sign(np.sin(theta)*np.cos(theta)) == np.sign(np.cos(phi_1-phi_2))).all()
    # mx = (np.sin(theta)*ft[:,0]+np.cos(theta)*ft[:,1])
    # mat = np.fft.ifft(mx, axis=0).real
    # assert (np.abs(mx)>=np.abs(ft.mean(-1))).all()

    # mag_xy = np.sqrt(mat_x**2 + mat_y**2)
    # mat = mag_xy * np.where(np.abs(mat_x) >= np.abs(mat_y), np.sign(mat_x), np.sign(mat_y))
    
    plt.plot(mat)
    plt.show()
    
    # import librosa
    n_mels = 100
    fmax = 4096
    fmin = 100
    sr = f
    y = mat
    y = y / np.abs(y).max()
    a = y
    fig, ax = plt.subplots(figsize=(12,5))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin, n_fft=8048)
    M_db = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax, fmax=fmax,fmin=fmin,sr=sr)
    plt.colorbar(img,ax=ax)
    plt.title('Speech Recovery (Ours)')
    plt.show()

    sf.write("abe_speech.wav", a, int(f))
    return a
# # %%
# %%time
# trip_flow = TripletMatchingFlowAlgorithm(width=w, height=h, radius = 3.0, dt_min=1, dt_max=int(1e6/1e4))
# out = compute_flow(trip_flow, batch_duration=1.0e6)
# out_der = process(out)
#%%
time_flow = TimeGradientFlowAlgorithm(w, h, radius = 7, min_flow_mag = 10.0, bit_cut = 0)
out = compute_flow(time_flow, batch_duration=12.0e6)
out_der = process(out)
#%%
# plane_flow = PlaneFittingFlowAlgorithm(w, h, radius = 3, normalized_flow_magnitude = 100, min_spatial_consistency_ratio = -1, max_spatial_consistency_ratio = -1, fitting_error_tolerance = -1, neighbor_sample_fitting_fraction = 0.30000001192092896)
# out = compute_flow(plane_flow, batch_duration=1.0e6)
# out_der = process(out)
#%%
# # sparse_flow = SparseOpticalFlowAlgorithm(w, h, tuning = SparseOpticalFlowAlgorithmParameters.Tuning.SlowObjects)
# sparse_flow = SparseOpticalFlowAlgorithm(
#     w, h,
#     distance_gain = 0.05000000074505806,
#     damping = 0.7070000171661377,
#     omega_cutoff = 7.0,
#     min_cluster_size = 7,
#     max_link_time = 100,
#     match_polarity = True,
#     use_simple_match = True,
#     full_square = True,
#     last_event_only = False,
#     size_threshold = 100000000
# )
# out = compute_flow(sparse_flow, batch_duration=1.0e6)
# out_der = process(out)
#%%
#now low pass derivative to 5khz, integrate, then high pass to 10hz
from scipy.signal import butter, lfilter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = lfilter(b, a, data)
    return y
from copy import deepcopy
out_der_ = deepcopy(out_der)
# out_der_ = butter_lowpass_filter(out_der_, 10000, f)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = np.cumsum(out_der_)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = butter_highpass_filter(out_der_, 100, f)
# assert np.isfinite(out_der_).all()
# mel(out_der_)
out_der_ = out_der_ / np.abs(out_der_).max()
#now resample out_der to gt
out_der_sr = librosa.resample(out_der_, orig_sr=f, target_sr=sr)
import noisereduce
# out_der_sr = noisereduce.reduce_noise(y=out_der_sr, sr=16000, time_constant_s=100)
gt = librosa.load("abe_speech_gt.mp3", sr=sr)[0]

out_der_sr = align(out_der_sr, gt)

from pesq import pesq
from pystoi import stoi
print("pesq is", pesq(sr, gt, out_der_sr, 'wb'), "stoi is", stoi(gt, out_der_sr, sr, extended=False))
sf.write("abe_speech_bag_hat.wav", out_der_sr, sr)
sf.write("abe_speech_gt_sr.wav", gt, sr)
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
# from scipy.signal import correlate
# gt = np.zeros(10000)
# gt[1000] = 1
# out_der_sr = np.zeros(10000)
# out_der_sr[2000] = 1
# #%%
# correlation = correlate(out_der_sr, gt, mode='full')
# lags = np.arange(-len(out_der_sr) + 1, len(gt))
# #abs peak
# peak_ind = np.argmax(np.abs(correlation))
# peak_lag = lags[peak_ind]
# if peak_lag < 0:
#     out_der_sr = np.pad(out_der_sr, (-peak_lag, 0))
#     out_der_sr = out_der_sr[:len(gt)]
# else:
#     out_der_sr = np.pad(out_der_sr, (0, peak_lag))
#     out_der_sr = out_der_sr[-len(gt):]
# print("Peak lag:", peak_lag)
# # %%

# %%
