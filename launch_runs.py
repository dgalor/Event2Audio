#%%
import os, subprocess, sys
from tqdm import tqdm
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--event_dir", type=str, help="Dir name where files have format <ground truth identifier> with an optional _<experiment identifier>, and finally a .npy extension.", default="EventRecordings",
)
parser.add_argument(
    "--gt_dir", type=str, help="Dir name where files have format <ground truth identifier>_gt.<wav or mp3>", default="GroundTruth",
)
parser.add_argument(
    "--out_dir", type=str, help="Dir name for output files", default="Output",
)
parser.add_argument(
    "--mode", type=str, help="offline or online", default="online",
)
args = parser.parse_args()

gt_dir = args.gt_dir
event_dir = args.event_dir
out_dir = args.out_dir
mode = args.mode

assert mode in ["online", "offline"], "Mode must be either 'online' or 'offline'."
assert os.path.exists(gt_dir), f"Ground truth directory {gt_dir} does not exist."
assert os.path.exists(event_dir), f"Event recordings directory {event_dir} does not exist"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

gts = {f.split("_")[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(("_gt.wav", "_gt.mp3"))}
runs = []
for f in os.listdir(event_dir):
    key = f[:-4].split("_")[0]  # Assuming the file name starts with the key
    runs.append({
        "gt": gts[key],
        "event": os.path.join(event_dir, f),
        "out": os.path.join(out_dir, f[:-4] + "_hat_" + mode + ".wav")
    })
#%%
def command(recording, gt, out_path):
    return f"python run_{mode}.py --event_path {recording} --gt_path {gt} --out_path {out_path}"
def existing_files():
    return [i.split("_hat")[0] for i in os.listdir(out_dir) if mode in i]
#%%
for run in tqdm(runs, leave=True):
    out_path, recording, gt = run["out"], run["event"], run["gt"]
    out_path_minimal = os.path.basename(out_path).split("_hat")[0]
    if out_path_minimal in existing_files():
        i = existing_files().index(out_path_minimal)
        print(f"File {out_path_minimal} already exists")
    else:
        print(f"Processing {recording}")
        subprocess.Popen(command(recording, gt, out_path), shell=True)
        while out_path_minimal not in existing_files():
            time.sleep(1)
# %%
