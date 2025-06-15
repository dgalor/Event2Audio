import numpy as np
import subprocess

# %%
# Define lists of parameters
start_times = 0.8
end_times = 4

time_gap = 0.3
start_times = np.arange(start_times, end_times, time_gap)
end_times = np.append(start_times[1:], end_times)
# %%
# Ensure both lists are of the same length
if len(start_times) != len(end_times):
    raise ValueError("start_times and end_times must be of the same length.")

# Iterate over the lists and call the second script
for start_time, end_time in zip(start_times, end_times):
    subprocess.run(
        [
            "python",
            "prod_main.py",
            "--start_time",
            str(start_time),
            "--end_time",
            str(end_time),
            "--file_name",
            "data/0509_npy/0509_npy_chirp/1.npy",
            "--save_dir",
            "output/0509/0509_npy_chirp/1",
        ]
    )

# %%
