#%%
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm
import threading
import numpy as np
from functools import reduce
import cv2
import time
import matplotlib.pyplot as plt
event_height, event_width = 720, 1280
color = ColorPalette(2) #gray

stop_event = threading.Event()
# %%

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    image = np.array(image)
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)

def event_image_means(event_arr, grid_shape=(event_height, event_width)):
    out = np.zeros(grid_shape)
    for j in event_arr:
        out[j["y"], j["x"]] += 1
    return out

def gen2(duration):
    iterator = EventsIterator(
        input_path="",
        delta_t=duration*1e6,
        max_duration=duration*1e6,
    )
    frame = np.zeros((event_height, event_width), dtype=np.float128)
    events = next(iter(iterator))
    for event in events:
        frame[event["y"], event["x"]] += 1
    frame = frame/duration
    frame = image_histogram_equalization(frame.max()-frame)
    return frame

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
        self.is_motion = False
    def set_frame(self, ts, frame):
        frame_change = np.abs((frame-self.frame)/self.frame.clip(1)).mean()
        self.frame = frame
        self.received_frame = False 
        if frame_change > 0.15:
            self.is_motion = True
            # self.generator.skip_frames_up_to(ts+5*10**5)
            self.generator.reset()
        else:
            self.is_motion = False
        # else:
        #     self.accumulation_time += self.original_accumulation_time
    def get_frame(self):
        self.received_frame = True
        return self.frame, self.is_motion
    def is_frame_available(self):
        return not self.received_frame
    def run(self):
        iterator = EventsIterator(
            input_path="",
            delta_t=1e6/self.batch_rate,
            max_duration=self.max_duration,
        )
        try:
            for event_slice in iterator:
                if stop_event.is_set():
                    break
                self.generator.process_events(event_slice)
        except (StopIteration, KeyboardInterrupt, Exception) as e:
            print("exception", e)
        print("end of iteration")
        del iterator
# %%
stop_event.clear()
fps = 30
duration = None
accumulation = 10
try:
    recorder = PeriodicRecorder(max_duration=duration, fps=fps, accumulation_time=accumulation, batch_rate=100)
    thread = threading.Thread(target=recorder.run)
    thread.start()
    timeout = 5
    start = time.time()
    while True:
        if recorder.is_frame_available():
            frame, ismotion = recorder.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.equalizeHist(frame.max()-frame)
            if ismotion:
                cv2.putText(frame, "Motion detected", (360, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 0, 0), 4, bottomLeftOrigin=False)
            cv2.imshow("frame", frame)
            
            end = time.time()
            print(end='\x1b[2K')
            print(f"FPS: {1/(end-start)}", end="\r")
            start = time.time()
        else:
            if timeout < time.time() - start:
                print("Timeout")
                break
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
except (StopIteration, KeyboardInterrupt, Exception) as e:
    print(e)
cv2.destroyAllWindows()
stop_event.set()
thread.join()
del recorder, thread
# %%