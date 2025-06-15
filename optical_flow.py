import cv2
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

def compute_optical_flow_parallel_dekel(hist_frames):
    """
    Compute optical flow between consecutive histogram frames using parallel processing.

    Parameters
    ----------
    hist_frames : list or ndarray
        List/array of histogram frames (time, height, width)
    """
    # hist_norm = (hist_frames * 255 / np.max(hist_frames)).astype(np.uint8)
    hist_norm = (
        (hist_frames - hist_frames.min()) * 255 / (hist_frames.max() - hist_frames.min())
    ).astype(np.uint8)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(hist_frames) - 1) as pbar:
            def calculate_flow(i):
                # Calculate optical flow for a pair of frames
                flow = cv2.calcOpticalFlowFarneback(
                    hist_norm[i], hist_norm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                w = np.abs(hist_frames[i]) + np.abs(hist_frames[i + 1])
                w = w / w.sum().clip(1e-6)
                pbar.update(1)
                return (w[..., None]*flow).sum((0, 1))
            results = [i for i in executor.map(calculate_flow, range(len(hist_frames) - 1))]
    return np.stack(results, axis=0)

def compute_optical_flow_parallel(hist_frames):
    """
    Compute optical flow between consecutive histogram frames using parallel processing.

    Parameters
    ----------
    hist_frames : list or ndarray
        List/array of histogram frames (time, height, width)

    Returns
    -------
    flows : list
        List of optical flow fields for each consecutive pair of frames
    magnitudes : list
        List of flow magnitude arrays
    angles : list
        List of flow angle arrays
    """
    flows = []
    magnitudes = []
    angles = []

    # Convert histogram data to numpy array and then to 8-bit format for OpenCV
    # hist_frames = np.array(hist_frames)

    # hist_norm = (hist_frames * 255 / np.max(hist_frames)).astype(np.uint8)
    hist_norm = (
        (hist_frames - hist_frames.min()) * 255 / (hist_frames.max() - hist_frames.min())
    ).astype(np.uint8)

    def calculate_flow(i):
        # Calculate optical flow for a pair of frames
        flow = cv2.calcOpticalFlowFarneback(
            hist_norm[i], hist_norm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return flow, magnitude, angle

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(calculate_flow, range(len(hist_frames) - 1)))

    # Unpack results
    for flow, magnitude, angle in results:
        flows.append(flow)
        magnitudes.append(magnitude)
        angles.append(angle)

    return flows, magnitudes, angles


def compute_optical_flow(hist_frames):
    """
    Compute optical flow between consecutive histogram frames.

    Parameters
    ----------
    hist_frames : list or ndarray
        List/array of histogram frames (time, height, width)

    Returns
    -------
    flows : list
        List of optical flow fields for each consecutive pair of frames
    magnitudes : list
        List of flow magnitude arrays
    angles : list
        List of flow angle arrays
    """
    flows = []
    magnitudes = []
    angles = []

    # Convert histogram data to numpy array and then to 8-bit format for OpenCV
    hist_frames = np.array(hist_frames)
    hist_norm = (hist_frames * 255 / np.max(hist_frames)).astype(np.uint8)

    for i in range(len(hist_frames) - 1):
        # Calculate optical flow
        # cv2.calcOpticalFlowFarneback is a function that calculates the optical flow for a pair of images using the Farneback method.
        # The parameters are:
        # - prev: the previous frame
        # - next: the next frame
        # - None: the mask of valid pixels, if None, all pixels are valid
        # - 0.5: the pyr_scale parameter, which is the ratio of the image scale between the two pyramid levels
        # - 3: the levels parameter, which is the number of pyramid levels
        # - 15: the winsize parameter, which is the size of the search window at each pyramid level
        # - 3: the iterations parameter, which is the number of iterations at each pyramid level
        # - 5: the poly_n parameter, which is the size of the pixel neighborhood used to find polynomial expansion in each pixel
        # - 1.2: the poly_sigma parameter, which is the standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
        # - 0: the flags parameter, which is the operation flags
        flow = cv2.calcOpticalFlowFarneback(
            hist_norm[i], hist_norm[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        flows.append(flow)
        magnitudes.append(magnitude)
        angles.append(angle)

    return flows, magnitudes, angles
