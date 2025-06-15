#%%
# %matplotlib widget
import numpy as np
from metavision_sdk_cv import TripletMatchingFlowAlgorithm, SparseOpticalFlowAlgorithm, TimeGradientFlowAlgorithm, PlaneFittingFlowAlgorithm
from tqdm import trange
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal
import librosa

sr = 16000
gt = librosa.load("abe_speech_gt.mp3", sr=sr)[0]

events = np.load("abe_speech_bag.npy")

s, e, dfactor = 3.0+0e-6, 3+2e-6, 1 #12.0

cond = (events["t"]>=s*1e6) & (events["t"]<=e*1e6)
events = events[cond][::dfactor]
events["t"] = events["t"] - events["t"].min()

e, s = events["t"].max()/1e6, events["t"].min()/1e6
events["x"] = events["x"]-events["x"].min()
events["y"] = events["y"]-events["y"].min()
w, h, lent = events["x"].max()+1, events["y"].max()+1, events["t"].max()

alg = TimeGradientFlowAlgorithm(w, h, radius = 1, min_flow_mag = 10.0, bit_cut = 0)
buffer = alg.get_empty_output_buffer()
alg.process_events(events, buffer)
flow = buffer.numpy().copy()

# tv, vx, vy = flow["t"], flow["vx"], flow["vy"]
# coord_tv = np.round((tv/1e6-s)*1e5).astype(np.uint32)
# counts = coo_matrix((np.ones_like(coord_tv), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# mat_x = coo_matrix((vx, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# mat_y = coo_matrix((vy, (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# # mat_amp_x = coo_matrix((np.abs(vx), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# # mat_amp_y = coo_matrix((np.abs(vy), (coord_tv,np.zeros_like(coord_tv))), shape=(coord_tv.max() + 1, 1)).toarray()[:, 0]
# mat_x = mat_x / counts.clip(1, None)
# mn, std = np.abs(mat_x).mean(), np.abs(mat_x).std()
# mat_x = np.clip(np.abs(mat_x), mn-5*std, mn+5*std) * np.sign(mat_x)
# mat_y = mat_y / counts.clip(1, None)
# mn, std = np.abs(mat_y).mean(), np.abs(mat_y).std()
# mat_y = np.clip(np.abs(mat_y), mn-5*std, mn+5*std) * np.sign(mat_y)
# mag = np.sqrt(mat_x**2 + mat_y**2)
# mat = mag * (np.sign(mat_y) if np.abs(mat_x).mean() < np.abs(mat_y).mean() else np.sign(mat_x))
#%%
#plot 3d scatterplot of events
np.random.seed(1)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
x, y, t = events["x"]/w, events["y"]/h, events["t"]/lent*10
i = events["t"] == 0
t = t + np.random.randn(len(t))
ax.scatter(t[i], x[i], y[i], c=events["p"][i], cmap='bwr', s=100, alpha=1.0)
ax.set_box_aspect((5, 1, 1)) 
plt.axis('off')
#%%
it = 0
fig = plt.figure(figsize=(5, 5))
plt.scatter(x[t==it], y[t==it], c=events["p"][t==it], cmap='bwr', s=100, alpha=1.0)
plt.axis("off")
#%%
# ax = fig.add_subplot(122, projection='3d')
# flowi = flow[(flow["t"]>900) & (flow["x"]%10==0) & (flow["y"]%10==0)]
# tv, xv, yv, vx, vy = flowi["t"]/lent*2, flowi["x"]/w, flowi["y"]/h, flowi["vx"], flowi["vy"]
# vx = vx / np.abs(vx).max()
# vy = vy / np.abs(vy).max()
# ax.quiver(tv, xv, yv, np.zeros_like(vx), vx, vy, color='black', arrow_length_ratio=0.2)
# plt.axis('off')
#%%
i = (x > 0.8) & (y > 0.8) & (t > 1.9)
xi, yi, ti = x[i], y[i], t[i]
pi = events["p"][i]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ti, xi, yi, c=pi, cmap='bwr')
#%%
#now simulate speckle with feature size of 10 by lowpassing over space 
np.random.seed(1)
nt = 10
speckle = np.random.randn(h+nt, w+nt)
speckle = signal.convolve2d(speckle, np.ones((10, 10))/100, mode='same')
speckle[speckle > 0] = 1
speckle[speckle <= 0] = -1
speckles = [np.roll(np.roll(speckle, i, axis=0), i, axis=1)[nt//2:-nt//2, nt//2:-nt//2] for i in range(nt)]
speckle = np.stack(speckles, axis=2)
speckle = speckle / np.abs(speckle).max()
#now show all the speckles in a grid
fig = plt.figure()
nrows = np.ceil(np.sqrt(nt)).astype(int)
ncols = np.ceil(nt/nrows).astype(int)
for i in range(nt):
    ax = fig.add_subplot(nrows, ncols, i+1)
    ax.imshow(speckle[:, :, i], cmap='bwr')
    ax.axis('off')

#psudoevents. first get 3d grid for speckle via meshgrid
xp, yp, tp = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), np.linspace(0, 2, nt))
eps=0.0#2
i = (np.in1d(xp.flatten(), np.linspace(0, 1, w)[::5])) & (np.in1d(yp.flatten(), np.linspace(0, 1, h)[::5]))

xp = xp.flatten() + np.random.randn(w*h*nt)*eps
yp = yp.flatten() + np.random.randn(w*h*nt)*eps
tp = tp.flatten() + np.random.randn(w*h*nt)*eps*2
pp = speckle.flatten()

vxp = np.ones_like(xp)
vyp = np.ones_like(yp)

eventsp = np.stack((xp, yp, tp, pp), axis=-1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tp, xp, yp, c=pp, cmap='bwr', s=1, alpha=1.0)
l = 0.3
origin = [-0.3, -0.25, 0]
ax.quiver(*origin, l, 0, 0, color='black', arrow_length_ratio=0.2)
ax.quiver(*origin, 0, l, 0, color='black', arrow_length_ratio=0.2)
ax.quiver(*origin, 0, 0, l, color='black', arrow_length_ratio=0.2)
# plt.axis('off')
# plt.savefig("speckle.png", dpi=300)
#pick the last event at center

#plot vector field but downsample x and y by 10
iouter = (xp < 0.2) & (yp < 0.2) & (tp > 1.5) & i
xpi, ypi, tpi = xp[iouter], yp[iouter], tp[iouter]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# eps = 0.005/w+eps*np.random.randn(len(xpi)), /h+eps*np.random.randn(len(xpi))
ax.quiver(2.1, 1.0, 0.0, 0.0, 1.0, 1.0, color='black', arrow_length_ratio=0.2)
# ax.quiver(tpi, xpi, ypi, np.ones_like(xpi), np.ones_like(xpi), np.ones_like(xpi), color='black', arrow_length_ratio=0.2)
# plt.axis('off')
# %%
np.random.seed(1)
nt = 20
speckle = np.random.randn(h+nt)
speckle = signal.convolve(speckle, np.ones(5)/5, mode='same')
speckle[speckle > 0] = 1
speckle[speckle <= 0] = -1
speckles = [np.roll(speckle, i, axis=0)[nt//2:-nt//2] for i in range(nt)]
speckle = np.stack(speckles, axis=-1)
speckle = speckle / np.abs(speckle).max()
#now show all the speckles in a grid
plt.imshow(speckle, cmap='bwr', origin='lower')
plt.show()

xp, tp = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 2, nt))
eps=0.03
xp = xp.flatten() + np.random.randn(w*nt)*eps
tp = tp.flatten() + np.random.randn(w*nt)*eps
pp = speckle.flatten()

vxp = np.ones_like(xp)

eventsp = np.stack((xp, tp, pp), axis=-1)
plt.scatter(tp, xp, c=pp, cmap='bwr', s=10, alpha=0.8)
plt.show()
# %%