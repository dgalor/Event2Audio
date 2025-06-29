#  Event2Audio: Event-Based Optical Vibration Sensing (Official Implementation)
### [Project Page](https://mingxuancai.github.io/event2audio/) | [Data](https://berkeley.box.com/s/4kdfmdx84xhot3145qkhnh1s2qg5w55e)

Small vibrations observed in video can unveil information beyond what is visual, such as sound and material properties. It is possible to passively record these vibrations when they are visually perceptible, or actively amplify their visual contribution with a laser beam when they are not perceptible.

In this project we develop a vibration sensing system that relies on event-based cameras. This emerging imaging modality utilizes asynchronous pixels that only report pixel-level brightness changes, generating a sparse stream of events. These features make event-based cameras highly efficient for capturing and processing fast motion.

This repo implements the two methods described in the paper, which reconstruct audio from speckle-generated (raw) events.

![Image](https://github.com/user-attachments/assets/493503b6-02e6-43a8-a5c1-d83b6fdc2222)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dgalor/Event2Audio.git
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

## Acknowledgements

Authors: Mingxuan Cai*, Dekel Galor*, Amit Kohli, Jacob Yates, Laura Waller

*Mingxuan Cai and Dekel Galor co-led this project with equal contribution.