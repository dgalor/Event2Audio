#%%
import numpy as np
from ALP4 import *
import time

# Load the Vialux .dll and initialize the device
DMD = ALP4(version='4.1', libDir='C:\\Users\\dekel\\Documents\\event-noise2image')
DMD.Initialize()

# Define ROI parameters (for example: use 100 rows starting at row 100)
roi_start = 100   # starting row of the ROI
roi_height = 100  # number of rows in the ROI
full_width = DMD.nSizeX  # we keep the full horizontal resolution

# Create binary amplitude images for the ROI only
bitDepth = 1
imgBlack_roi = np.zeros((roi_height, full_width), dtype=np.uint8)
imgWhite_roi = np.ones((roi_height, full_width), dtype=np.uint8) * (2**8 - 1)

# Concatenate the two images into one sequence (flattened)
imgSeq_roi = np.concatenate([imgBlack_roi.ravel(), imgWhite_roi.ravel()])

# Allocate onboard memory for a sequence of 2 images
DMD.SeqAlloc(nbImg=2, bitDepth=bitDepth)

# Use SeqPutEx to send only the ROI data:
DMD.SeqPutEx(imgData=imgSeq_roi, LineOffset=roi_start, LineLoad=roi_height)

# Optionally, if your API requires setting the active area, you can use:
# DMD.SeqControl(ALP_SEQ_DMD_LINES, MAKELONG(roi_start, roi_height))
# (Uncomment the above if needed.)

# Set image rate (here using pictureTime)
f = 4000
t = int(1e6 / f)
DMD.SetTiming(pictureTime=t)

# Run the sequence in an infinite loop
DMD.Run()
time.sleep(3)

# Stop the sequence display and clean up
DMD.Halt()
DMD.FreeSeq()
DMD.Free()
