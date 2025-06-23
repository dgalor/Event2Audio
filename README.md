# Event2Audio: Event-Based Optical Vibration Sensing

This project reconstructs audio from video data captured by an event camera. It uses optical flow to analyze the vibrations of objects in the video and converts these vibrations back into sound.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dgalor/event-visual-vibrometry.git Event2Audio
   cd Event2Audio
   ```

2. **Install dependencies:**
   Install the metavision sdk for python (not required for offline algorithm) from https://docs.prophesee.ai/4.6.2/index.html
   
   Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Download our event data**
   https://berkeley.box.com/s/4kdfmdx84xhot3145qkhnh1s2qg5w55e
## Usage

There are currently three main scripts.

### Offline Processing

To process a pre-recorded event data file (`.npy`) and reconstruct the audio using the _offline_ algorithm, use `run_offline.py`.

```bash
python run_offline.py --event_path EventRecordings/path.npy --gt_path GroundTruth/path.wav --out_path Output/path.wav
```

### Online Processing

To process a pre-recorded event data file (`.npy`) and reconstruct the audio using the _realtime_ algorithm, use `run_online.py`.

```bash
python run_online.py --event_path EventRecordings/path.npy --gt_path GroundTruth/path.wav --out_path Output/path.wav
```

### Launch runs for a folder of recordings

The script `launch_runs.py` enables processing a folder of recordings with either the _realtime_ or _offline_ algorithm.

```bash
python launch_runs.py --event_dir EventRecordings --gt_dir GroundTruth --out_dir Output --mode online
```

## Acknowledgments

Authors: Mingxuan Cai*, Dekel Galor*, Amit Kohli, Jacob Yates, Laura Waller

*Mingxuan Cai and Dekel Galor co-led this project with equal contribution.