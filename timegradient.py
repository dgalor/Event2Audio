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

batch = np.array([
    # (10, 10, 0, 10),
    # (10, 10, 0, 31),
    (16, 10, 0, 37),
    # (13, 10, 0, 30),
    (13, 10, 0, 50),
    # (7, 10, 0, 8),
    # (10, 7, 0, 8),
    # (10, 13, 0, 8)
], dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')])
# batch=events[:100000]
alg = TimeGradientFlowAlgorithm(100, 100, radius = 3, min_flow_mag = 10.0, bit_cut = 0)
t1 = time.time()
buffer = alg.get_empty_output_buffer()
alg.process_events(batch, buffer)
print("Time taken:", time.time()-t1)
buffer.numpy().copy()
# %%
