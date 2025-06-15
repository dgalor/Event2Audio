#%%
import os, subprocess
from tqdm import tqdm
import time
overwrite = True
gts = [
    "data/gt/abe_speech.mp3",
    "data/gt/abe_speech.mp3",
    "data/gt/japanese_gt.wav",
    "data/gt/noisy_gt.wav",
    "data/gt/woman2_echo_gt.wav",
]
recordings = [
    "data/abspeech_chipbag.npy",
    "data/abspeech_water.npy",
    "data/japanese1.npy",
    "data/noise2.npy",
    "data/woman2_echo.npy",
]
post = "_noPCA"
out_paths = [
    f"data/hat/abspeech_chipbag_online_hat{post}.wav",
    f"data/hat/abspeech_water_online_hat{post}.wav",
    f"data/hat/japanese1_online_hat{post}.wav",
    f"data/hat/noise2_online_hat{post}.wav",
    f"data/hat/woman2_echo_online_hat{post}.wav",
]
def command(recording, gt, out_path):
    return f"python run_online.py --data_path {recording} --gt_path {gt} --out_path {out_path}"
def existing_files():
    return [i.split("_pesq")[0]+".wav" for i in os.listdir("data/hat/")]
#%%
for recording, gt, out_path in tqdm(zip(recordings, gts, out_paths), leave=True):
    out_path_minimal = out_path.split("/")[-1]
    if out_path_minimal in existing_files():
        i = existing_files().index(out_path_minimal)
        if overwrite:
            os.remove('data/hat/'+list(os.listdir('data/hat/'))[i])
        else:
            print(f"File {list(os.listdir('data/hat/'))[i]} already exists")
            continue
    print(f"Processing {recording}")
    subprocess.Popen(command(recording, gt, out_path), shell=True)
    while out_path_minimal not in existing_files():
        time.sleep(1)
# %%
