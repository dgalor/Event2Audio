#%%
import numpy as np
from ALP4 import *
import time

# Load the Vialux .dll
DMD = ALP4(version = '4.1', libDir='C:\\Users\\dekel\\Documents\\event-noise2image')
# Initialize the device
DMD.Initialize()

# Binary amplitude image (0 or 1)
bitDepth = 1    
imgBlack = np.zeros([DMD.nSizeY,DMD.nSizeX])
imgWhite = np.ones([DMD.nSizeY,DMD.nSizeX])*(2**8-1)
imgSeq  = np.concatenate([imgBlack.ravel(),imgWhite.ravel()])

# Allocate the onboard memory for the image sequence
DMD.SeqAlloc(nbImg = 2, bitDepth = bitDepth)
# Send the image sequence as a 1D list/array/numpy array
DMD.SeqPut(imgData = imgSeq)
# Set image rate to 50 Hz
f = 4000
t = int(1e6/f)
DMD.SetTiming(pictureTime = t)#20000//200)

# Run the sequence in an infinite loop
DMD.Run()

time.sleep(3)

# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()
# %%
