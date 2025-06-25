# from utils import process_audio_file, process_audiào_file_super_fast
import os
import numpy as np
import cv2
from scipy.signal import spectrogram
from scipy.signal.windows import blackman
from matplotlib import pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
from scipy.io import wavfile
import time
from predict_and_extract_online import prepare_image_batch, ProcessingConfig
from prepare_audio import process_audio_file_super_fast2, process_audio_file_old

def process_audio_file_corrected(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=3, 
                       save=False, wlen=2048, nfft=2048, sliding_w=0.4, cut_low_frequency=3, 
                       cut_high_frequency=20, target_width_px=1167, target_height_px=875):
    """
    Process an audio file and generate spectrogram images.

    This optimized version avoids creating matplotlib figures for every spectrogram. 
    Instead, it converts the computed spectrogram (in dB) directly to a grayscale image,
    crops the frequency range, normalizes the values, resizes using OpenCV, and finally
    saves the image if requested.

    Parameters:
        file_path (str): Path to the audio file.
        saving_folder (str): Folder to save images (if save=True).
        batch_size (int): Number of spectrogram images to generate.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        save (bool): Whether to save the images.
        wlen (int): Window length for spectrogram calculation.
        nfft (int): Number of FFT points.
        sliding_w (float): Duration of each slice in seconds.
        cut_low_frequency (int): Lower frequency limit (in kHz) for the spectrogram.
        cut_high_frequency (int): Upper frequency limit (in kHz) for the spectrogram.
        target_width_px (int): Target image width in pixels.
        target_height_px (int): Target image height in pixels.

    Returns:
        images (list): List of spectrogram images as numpy arrays.
        
    Raises:
        FileNotFoundError: If the audio file is not found.
    """
    import os
    import numpy as np
    import cv2
    from scipy.signal import spectrogram
    # Use NumPy’s Blackman window; alternatively, you can import from scipy.signal.windows
    win = blackman(wlen, sym=False)
    hop = round(0.8 * wlen)  # window hop size
    # Check file type and read accordingly
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.flac':
        # try:
        #     import soundfile as sf
        #     x, fs = sf.read(file_path)
        # except ImportError:
            # Fallback to librosa if soundfile is not available
        try:
            import librosa
            x, fs = librosa.load(file_path, sr=None)  # sr=None preserves original sample rate
        except ImportError:
            raise ImportError("Please install soundfile or librosa to process FLAC files.")
    else:
        try:
            from scipy.io import wavfile
            fs, x = wavfile.read(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found.")
    
    # Create saving folder if saving is enabled
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    N = len(x)
    if end_time is not None:
        N = min(N, int(end_time * fs))
    low = int(start_time * fs)
    new_samples_per_slice = int(sliding_w * fs)
    samples_per_slice = int(0.8 * fs)

    # Pre-calculate frequency cropping indices later using the frequency array (f) from the first slice
    first_slice = True
    for _ in range(batch_size):
        if low + samples_per_slice > N:
            break
        
        x_w = x[low:low + samples_per_slice]
        win = blackman(wlen, sym=False)
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=wlen-hop, nfft=nfft, window=win, scaling='density', mode='psd')
        # Convert to dB scale as in original
        Sxx = 10 * np.log10(np.abs(Sxx) + 1e-19)
        # Normalize Sxx to 0-255
        Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx)) * 255

        if first_slice:
            # f is in Hz; use kHz limits
            low_freq_hz = cut_low_frequency * 1000
            high_freq_hz = cut_high_frequency * 1000
            low_idx = np.searchsorted(f, low_freq_hz)
            high_idx = np.searchsorted(f, high_freq_hz)
            first_slice = False
        
        # Crop frequency axis as original (note: original divides f by 1000 for plotting,
        # but we use the indices determined from the Hz values)
        Sxx_cropped = Sxx[low_idx:high_idx, :]
        
        # reverse the y axis 
        Sxx_cropped = np.flipud(Sxx_cropped)

        # Normalize the spectrogram to 0-255 grayscale
        Sxx_cropped_uint8 = np.clip(Sxx_cropped, 0, 255).astype(np.uint8)
        # Convert to grayscale image (already grayscale, no need for color conversion if Sxx is 2D)
        img_gray = Sxx_cropped_uint8

        # Resize image to target dimensions; using INTER_LINEAR for smoother interpolation
        resized =  cv2.resize(img_gray, (target_width_px, target_height_px), interpolation=cv2.INTER_NEAREST)
        # Convert to 3-channel image by stacking the grayscale image three times
        image = np.stack([resized, resized, resized], axis=2)
        
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs:.2f}_fast.jpg")
            cv2.imwrite(image_name, image)
        
        images.append(image)
        low += new_samples_per_slice

    return images



def process_audio_file_local(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
    """
    Process an audio file and generate spectrogram images.

    Parameters:
    - file_path (str): Path to the audio file.
    - saving_folder (str): Path to the folder where the spectrogram images will be saved. Default is "./images".
    - batch_size (int): Number of spectrogram images to generate. Default is 50.
    - start_time (float): Start time in seconds for processing the audio file. Default is 0.
    - end_time (float): End time in seconds for processing the audio file. Default is None.
    - save (bool): Whether to save the spectrogram images. Default is False.
    - wlen (int): Length of the window for spectrogram calculation. Default is 2048.
    - nfft (int): Number of points for FFT calculation. Default is 2048.
    - sliding_w (float): Sliding window size in seconds. Default is 0.4.
    - cut_low_frequency (int): Lower frequency limit for the spectrogram plot. Default is 3.
    - cut_high_frequency (int): Upper frequency limit for the spectrogram plot. Default is 20.
    - target_width_px (int): Width of the spectrogram image in pixels. Default is 903.
    - target_height_px (int): Height of the spectrogram image in pixels. Default is 677.

    Returns:
    - images (list): List of spectrogram images as numpy arrays.

    Raises:
    - FileNotFoundError: If the audio file is not found.
    """

    try:
        # Load sound recording
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Create the saving folder if it doesn't exist
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    # Calculate the spectrogram parameters
    hop = round(0.8 * wlen)  # window hop size
    win = blackman(wlen, sym=False)

    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    N = len(x)  # signal length

    if end_time is not None:
        N = min(N, int(end_time * fs))

    low = int(start_time * fs)
    
    samples_per_slice = int(sliding_w * fs)

    for _ in range(batch_size):
        if low + samples_per_slice > N:  # Check if the slice exceeds the signal length
            break
        x_w = x[low:low + samples_per_slice]
        
        # Calculate the spectrogram
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=wlen-hop, nfft=wlen*2, window=win, scaling='density',  # Power spectral density
        mode='psd')
        Sxx = 10 * np.log10(np.abs(Sxx)+1e-14)  # Convert to dB
        
        # Cut the frequencies directly in the data
        low_idx = np.searchsorted(f / 1000, cut_low_frequency)
        high_idx = np.searchsorted(f / 1000, cut_high_frequency)
        f_cut = f[low_idx:high_idx]
        Sxx_cut = Sxx[low_idx:high_idx, :]
        
        # Create the spectrogram plot
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f_cut / 1000, Sxx_cut, cmap='gray')
        # No need for ylim as we've already cut the data

        ax.set_axis_off()  # Turn off axis
        # fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])
        
        # Adjust margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
      
        # Save the spectrogram as a JPG image without borders
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
            fig.savefig(image_name, dpi=plt.rcParams['figure.dpi'])  # Save without borders

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        low += samples_per_slice//2

    plt.close('all')  # Close all figures to release memory

    return images


config = ProcessingConfig(
        batch_duration=None,
        batch_size=2,
        cut_low_frequency=None,
        cut_high_frequency=None,
        image_normalize=False,
        image_size=(224, 224),  # Standard size for most models
        save_positive_examples = True,
    )

# wavtest = "/home/emanuelli/Téléchargements/68028001.wav"
flactest = "/home/emanuelli/Bureau/Test_material/2021_06_24_08_44_07_10.flac"

def test_process(function, *args):
    start = time.time()
    images = function(flactest, saving_folder= "./tests_with_processaudio", 
        batch_size=config.batch_size, start_time=39.2, end_time=None, save=True, wlen=2048, 
        nfft=2048, target_height_px=677, target_width_px=902, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20)

    _ , batch_info = prepare_image_batch(images, start_time = 0, config = config)
    non_prep_image = images[0]
    prepared_images = batch_info[0]

    # print(prepared_images)
    # print(non_prep_image==prepared_images[0])

    cv2.imwrite(f"testeee_{function}.jpg", prepared_images[0])
    print(f"Execution time: {time.time() - start:.2f} seconds")
    return prepared_images[0]

test_process(process_audio_file_corrected)

# print(test_process(process_audio_file_local) == test_process(process_audio_file))

# print( test_process(process_audio_file_local)== test_process(process_audio_file_super_fast))

# print( test_process(process_audio_file_super_fast)== test_process(process_audio_file_super_fast2))


# test_process(process_audio_file_local)





# def prepare_image_batch(images: List[np.ndarray], start_time: float, 
#                                   config: ProcessingConfig) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float]]]:
#     """
#     Vectorized implementation of image batch preparation for massive performance improvement.
    
#     Args:
#         images: List of spectrograms
#         start_time: Start time of the first image
#         config: Processing configuration
        
#     Returns:
#         Tuple of processed images ready for prediction and time information
#     """
#     if not images:
#         return np.array([]), []
    
#     # Pre-allocate arrays with exact shape needed
#     batch_size = len(images)
    
#     # Stack images first for vectorized operations
#     # Use numpy's stack to avoid Python loop
#     original_images = [img.copy() for img in images]
    
#     # Calculate all start/end times at once using vectorized operations
#     image_start_times = start_time + np.arange(batch_size) * 0.4
#     image_end_times = image_start_times + 0.4
    
#     # Round for consistent output
#     image_start_times = np.round(image_start_times, 2)
#     image_end_times = np.round(image_end_times, 2)
    
#     # Create time info list
#     time_info = [(original_images[i], image_start_times[i], image_end_times[i]) 
#                  for i in range(batch_size)]
    
#     # Process all images at once when possible
#     # Convert to numpy array for vectorized operations
#     processed_images = []
    
#     # Use OpenCV's batch processing if available
#     if hasattr(cv2, 'resize_batch'):
#         # OpenCV 4.7+ has batch processing
#         processed_images = cv2.resize_batch(images, config.image_size)
#     else:
#         # Fall back to list comprehension which is still faster than for loop
#         processed_images = np.array([cv2.resize(img, config.image_size) for img in images])
    
#     # Add channel dimension if needed - vectorized operation
#     if processed_images.shape[-1] != 3:
#         if len(processed_images.shape) == 3:  # [batch, height, width]
#             processed_images = np.repeat(processed_images[:, :, :, np.newaxis], 3, axis=3)
#         else:  # Handle single image case
#             processed_images = np.repeat(processed_images[:, :, np.newaxis], 3, axis=2)
    
#     # Normalize in a single vectorized operation if required
#     if config.image_normalize:
#         processed_images = processed_images / 255.0
    
#     # Ensure correct dtype for TensorFlow
#     return processed_images.astype(np.float32), time_info




# image path 
matlab_image = "/home/emanuelli/Bureau/Test_material/original images/2021_06_24_08_44_07_10.flac-39.6.jpg"
fast_image = "/home/emanuelli/Documents/GitHub/Dolphins/tests_with_processaudio/2021_06_24_08_44_07_10-39.60_fast.jpg"

for image in [fast_image, matlab_image]:
    # Load the image
    img = cv2.imread(image)
    # calculate the statistics on the pixel values
    # first, the mean
    mean = cv2.mean(img)[:3]
    # then the standard deviation
    stddev = cv2.meanStdDev(img)[1].flatten()[:3]
    # Print the results
    print("Mean:", mean)
    print("Standard Deviation:", stddev)
