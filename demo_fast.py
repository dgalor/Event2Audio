#%%
from metavision_core.event_io import EventsIterator
import metavision_hal as mv_hal
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm
import threading
import numpy as np
from functools import reduce
import cv2
import time
import matplotlib.pyplot as plt
import scipy

#SET ROI
x_start, y_start, width, height = 100, 100, 100, 100

#SET BIASES
biases = {
    'bias_diff': 80,
    'bias_diff_off': 52,
    'bias_diff_on': 115,
    'bias_fo': 74,
    'bias_hpf': 0,
    'bias_refr': 68
}

#%%
event_height, event_width = 720, 1280
color = ColorPalette(2) #gray

stop_event = threading.Event()
#%%
class PeriodicRecorder:
    def __init__(self, max_duration, accumulation_time=1, fps=30, batch_rate=1):
        accumulation_time = int(accumulation_time*10**6)
        max_duration = int(max_duration*10**6) if max_duration is not None else None
        self.frame = np.zeros((event_height, event_width, 3), dtype=np.uint8)
        self.generator = PeriodicFrameGenerationAlgorithm(event_width, event_height, accumulation_time, fps)
        self.generator.set_output_callback(self.set_frame)
        self.batch_rate = batch_rate
        self.max_duration = max_duration
        self.received_frame = False
        self.accumulation_time = accumulation_time
        self.original_accumulation_time = accumulation_time
        self.resetting = False
        self.roi = None
    def set_frame(self, ts, frame):
        self.frame = frame
        self.received_frame = False 
    def reset(self):
        self.resetting = True
    def get_frame(self):
        self.received_frame = True
        return self.frame
    def is_frame_available(self):
        return not self.received_frame
    def run(self):
        device = mv_hal.DeviceDiscovery.open(serial="") # opens the first available camera
        success = device.get_i_roi().set_mode(mv_hal.I_ROI.Mode(0))
        if success:
            wind_obj = device.get_i_roi().Window(x_start, y_start, width, height)
            device.get_i_roi().set_window(wind_obj)
            device.get_i_roi().enable(True)
        else:
            print("Failed to set roi")

        for bname, bval in biases.items():
            if not device.get_i_ll_biases().set(bname, bval):
                print("Failed to set biases")


        iterator = EventsIterator.from_device(
            device=device,
            delta_t=1e6/self.batch_rate,
            max_duration=self.max_duration,
        )
        # iterator = EventsIterator(
        #     input_path="",
        #     delta_t=1e6/self.batch_rate,
        #     max_duration=self.max_duration,
        # )
        try:
            for event_slice in iterator:
                if stop_event.is_set():
                    break
                elif self.resetting:
                    break
                self.generator.process_events(event_slice)
        except (StopIteration, KeyboardInterrupt, Exception) as e:
            print("exception", e)
        del iterator, device
        if self.resetting:
            print("resetting")
            self.generator.reset()
            self.resetting = False
            self.run()
        else:
            print("end of iteration")
# %%
stop_event.clear()
fps = 1000 #display rate - slow-mo if high since we can only display ~30fps
duration = None
accumulation = 1/fps #exposure per frame
blur = 0
try:
    recorder = PeriodicRecorder(max_duration=duration, fps=fps, accumulation_time=accumulation, batch_rate=100)
    thread = threading.Thread(target=recorder.run)
    thread.start()
    timeout = 5
    start = time.time()
    while True:
        if recorder.is_frame_available():
            frame = recorder.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # if ismotion:
            #     cv2.putText(frame, "Motion detected", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 0, 0), 4, bottomLeftOrigin=False)
            cv2.imshow("frame", frame)
            
            end = time.time()
            print(end='\x1b[2K')
            print(f"FPS: {1/(end-start)}", end="\r")
            start = time.time()
        else:
            if timeout < time.time() - start:
                print("Timeout")
                break
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting")
            recorder.reset()
            while recorder.resetting:
                pass
            print("Reset")
        #check if its a digit, and set blur kernel to it
        # elif key in [ord(str(i)) for i in range(10)]:
        #     # print(key, chr(key), key-48, "/n")
        #     blur = int(chr(key))
        #     print(f"Set blur kernel to {blur}")
except (StopIteration, KeyboardInterrupt, Exception) as e:
    print(e)
cv2.destroyAllWindows()
stop_event.set()
thread.join()
del recorder, thread
# %%