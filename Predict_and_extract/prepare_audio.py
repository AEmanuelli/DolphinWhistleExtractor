# Imports 

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

config = ProcessingConfig(
        batch_duration=None,
        batch_size=None,
        cut_low_frequency=None,
        cut_high_frequency=None,
        image_normalize=False,
        image_size=(224, 224),  # Standard size for most models
        save_positive_examples = False,
    )

def process_audio_file(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
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
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=hop, nfft=nfft, window=win)
        Sxx = 20 * np.log10(np.abs(Sxx)+1e-14)  # Convert to dB

        # Create the spectrogram plot
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f / 1000, Sxx, cmap='gray')
        ax.set_ylim(cut_low_frequency, cut_high_frequency)

        ax.set_axis_off()  # Turn off axis
        # fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])
        
        # Adjust margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        

        fig.savefig(f"{file_name}-{low/fs}.jpg", bbox_inches='tight', pad_inches=0, dpi=plt.rcParams['figure.dpi'])
        # Save the spectrogram as a JPG image without borders
        if save:
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs}.jpg")
            fig.savefig(image_name, dpi=plt.rcParams['figure.dpi'])  # Save without borders
            

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)  # Keep this line for now (if you need the expanded dimension later)
        from tensorflow.keras.applications.vgg16 import preprocess_input 
        image = preprocess_input(image)  # Preprocess the image
        print("Image shape BEFORE squeeze:", image.shape)  # Print BEFORE squeeze

        image = np.squeeze(image, axis=0)  # **THIS IS THE IMPORTANT LINE - MAKE SURE IT'S HERE**

        print("Image shape AFTER squeeze:", image.shape)   # Print AFTER squeeze

        print("Image dtype before cv2.imwrite:", image.dtype)
        print("Image min/max values:", image.min(), image.max())
        print("Saving image to:", f"image_{low/fs}.jpg") # Added filepath print

        # save the image
        cv2.imwrite(f"image_{low/fs}.jpg", image)
        images.append(image)

        low += samples_per_slice

    plt.close('all')  # Close all figures to release memory

    return images

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.signal.windows import blackman
import os


def process_audio_file_optimized_adapted(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,
                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
                       target_height_px=677):
    """
    Optimized spectrogram generation adapted to match the input and output of process_audio_file.

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
        # Load sound recording efficiently
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")

    # Convert to single precision and ensure mono
    x = x.astype(np.single)
    if x.ndim > 1:
        x = x[:, 0]  # Take first channel if stereo

    # Spectrogram parameters matching MATLAB
    hop = round(0.8 * wlen)  # window hop size
    win = blackman(wlen, sym=False)  # Blackman window

    # File naming
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    N = len(x)

    if end_time is not None:
        N = min(N, int(end_time * fs))

    low = int(start_time * fs)
    slide_samples = round(sliding_w * fs)
    images = []

    # Matplotlib setup to minimize overhead
    plt.ioff()  # Turn off interactive mode

    # Create the saving folder if it doesn't exist and saving is enabled
    if save:
        os.makedirs(saving_folder, exist_ok=True)

    for _ in range(batch_size):
        if low + slide_samples > N:
            break

        # Extract window
        x_w = x[low:low + slide_samples]

        # Compute spectrogram with exact MATLAB-like parameters
        f, t, Sg = spectrogram(x_w, fs, window=win, nperseg=wlen,
                               noverlap=wlen - hop, nfft=nfft,
                               mode='magnitude')

        # Convert frequency to kHz
        f = f / 1000

        # Convert to power in dB (add tiny epsilon to prevent log(0))
        Sg_db = 20 * np.log10(np.abs(Sg) + 1e-10)

        # Create figure with pre-set size to match exactly
        fig, ax = plt.subplots(figsize=(target_width_px / 100, target_height_px / 100), dpi=1000) # Adjusted figsize to use target_px and dpi

        # Exact reproduction of MATLAB's imagesc behavior
        im = ax.imshow(Sg_db, aspect='auto', origin='lower',
                       extent=[t[0], t[-1], f[0], f[-1]],
                       cmap='gray')

        # Set frequency limits precisely
        ax.set_ylim(cut_low_frequency, cut_high_frequency)

        # Remove ticks and margins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off() # Turn off axis like in original function

        # Remove whitespace completely
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_visible(False)
        ax.patch.set_visible(False)


        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)


        # Save image with pixel-perfect settings
        if save:
            save_path = os.path.join(saving_folder,
                                     f'{file_name}-{low/fs:.1f}.jpg') # Using start time as file_name_ex
            plt.savefig(save_path,
                        bbox_inches='tight',
                        pad_inches=0,
                        dpi=100)

        # Close figure to free memory
        plt.close(fig)

        # Update sliding window
        low += slide_samples

    plt.close('all')  # Ensure all figures are closed at the end
    return images


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
        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=wlen-hop, nfft=nfft, window=win)
        # Convert to dB scale as in original
        Sxx = 10 * np.log10(np.abs(Sxx) + 1e-14)
        
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
        Sxx_cropped = Sxx_cropped[::-1, :]
        
        # Mimic pcolormesh default normalization (per-slice dynamic range)
        vmin = Sxx_cropped.min()
        vmax = Sxx_cropped.max()
        # Prevent division by zero in flat spectra
        norm = (Sxx_cropped - vmin) / (vmax - vmin + 1e-14)
        # Map to 0-255 grayscale
        img_gray = np.uint8(255 * norm)
        
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



def process_audio_file_super_fast2(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, 
                                  save=False, wlen=2048, nfft=2048, sliding_w=0.4, cut_low_frequency=3, 
                                  cut_high_frequency=20, target_width_px=903, target_height_px=677):
    """
    Process an audio file and generate spectrogram images matching MATLAB's output.
    
    This version reproduces the MATLAB logic exactly:
      - Uses a periodic Blackman window.
      - Uses a noverlap = wlen - round(0.8*wlen) to match MATLAB's spectrogram call.
      - Converts the spectrogram to dB (20*log10(|S|)).
      - Crops the frequency axis to the same kHz limits.
    
    OpenCV is used for image resizing and saving (avoiding matplotlib for speed).
    
    Parameters:
        file_path (str): Path to the audio file.
        saving_folder (str): Folder to save images (if save=True).
        batch_size (int): Maximum number of spectrogram images to generate.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        save (bool): Whether to save the images.
        wlen (int): Window length for spectrogram calculation.
        nfft (int): Number of FFT points.
        sliding_w (float): Duration (in seconds) of each spectrogram slice.
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
    from scipy.signal import spectrogram, windows
    from scipy.io import wavfile

    # Generate periodic Blackman window (matching MATLAB's 'periodic' option)
    win = windows.blackman(wlen, sym=False)
    # In MATLAB: hop = round(0.8*wlen) so that noverlap = wlen - hop.
    hop = round(0.8 * wlen)
    noverlap = wlen - hop

    # Read audio file
    try:
        fs, x = wavfile.read(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found.")
    
    # Ensure single precision (float32)
    x = x.astype(np.float32)
    
    # Create saving folder if saving is enabled
    if save and not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    
    images = []
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    N = len(x)
    if end_time is not None:
        N = min(N, int(end_time * fs))
    low = int(start_time * fs)
    samples_per_slice = int(sliding_w * fs)
    
    # Determine cropping indices from first spectrogram slice
    first_slice = True
    for _ in range(batch_size):
        if low + samples_per_slice > N:
            break
        
        # Extract slice of audio
        x_w = x[low:low + samples_per_slice]
        # Compute spectrogram with the correct overlap and window
        
        
        # Convert to dB scale (adding a small constant to avoid log(0))
        Sxx_dB = 20 * np.log10(np.abs(Sxx) + 1e-14)
        
        if first_slice:
            # f is in Hz; use kHz limits as in MATLAB (fg = f/1000 then ylim([cut_low_frequency, cut_high_frequency]))
            low_freq_hz = cut_low_frequency * 1000
            high_freq_hz = cut_high_frequency * 1000
            # Find indices that match the frequency limits (exact match as possible)
            low_idx = np.searchsorted(f, low_freq_hz)
            high_idx = np.searchsorted(f, high_freq_hz)
            first_slice = False
        
        # Crop frequency axis to exactly match MATLAB's display
        Sxx_cropped = Sxx_dB[low_idx:high_idx, :]
        
        # Normalize per slice (MATLAB imagesc auto-scales to the data range)
        vmin = Sxx_cropped.min()
        vmax = Sxx_cropped.max()
        norm = (Sxx_cropped - vmin) / (vmax - vmin + 1e-14)
        img_gray = np.uint8(255 * norm)
        
        # Resize image if target dimensions are specified
        if target_width_px and target_height_px:
            resized = cv2.resize(img_gray, (target_width_px, target_height_px), interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            resized = img_gray
        
        # Stack to 3 channels to mimic MATLAB's JPEG saving of an RGB image (even though it is grayscale)
        image = np.stack([resized, resized, resized], axis=2)
        
        if save:
            # The file name includes the start time (in seconds) for this slice
            image_name = os.path.join(saving_folder, f"{file_name}-{low/fs:.2f}_fast.jpg")
            cv2.imwrite(image_name, image)
        
        images.append(image)
        low += samples_per_slice

    return images


def compare_processing_functions(file_path, **kwargs):
    """
    Compare the performance and output of the two audio processing functions.
    
    Parameters:
    - file_path (str): Path to the audio file to process
    - **kwargs: Additional arguments to pass to both processing functions
    
    Returns:
    - dict: Dictionary containing comparison results
    """
    import time
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import cv2
    results = {}
    # from predict_and_extract_online import prepare_image_batch
    # Measure execution time for standard function
    start_time = time.time()
    standard_images = process_audio_file(file_path, save=True, **kwargs)
    _, standard_images = prepare_image_batch(standard_images, start_time=0, config=config)
    standard_images = standard_images[0]
    standard_time = time.time() - start_time
    
    # Measure execution time for fast function
    start_time = time.time()
    fast_images = process_audio_file_optimized_adapted(file_path, save=False, **kwargs)
    _, fast_images  = prepare_image_batch(fast_images, start_time=0, config=config)
    fast_images = fast_images[0]
    fast_time = time.time() - start_time


    # compare the two fast functions
    start_time = time.time()
    fim = process_audio_file_corrected(file_path, save=False, **kwargs)
    _, prep_fast= prepare_image_batch(fim, start_time=0, config=config)
    fim = prep_fast[0]
    fast_time_ = time.time() - start_time   

    start_time = time.time()
    fim2 = process_audio_file(file_path, save=False, **kwargs)
    _, prep_fast2 = prepare_image_batch(fim2, start_time=0, config=config)
    fim2 = prep_fast2[0]
    fast_time2 = time.time() - start_time

    print(f"Fast function 1: {fast_time_:.2f}s")
    print(f"Fast function 2: {fast_time2:.2f}s")
    print(f"Speedup: {fast_time_ / fast_time2:.2f}x faster")



    standard_time = fast_time_
    fast_time = fast_time2

    standard_images = fim
    fast_images = fim2
    
    # Record time results
    results['standard_time'] = standard_time
    results['fast_time'] = fast_time
    results['speedup_factor'] = standard_time / fast_time if fast_time > 0 else float('inf')
    
    # Compare output counts
    results['standard_image_count'] = len(standard_images)
    results['fast_image_count'] = len(fast_images)
    
    # Compare image similarity if both functions produced images
    if standard_images and fast_images:
        # Create a directory for comparison images
        comparison_dir = os.path.join(kwargs.get('saving_folder', './compare processing'), 'comparisons')
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)
        
        # Flatten batches if necessary
        flat_std_images = []
        flat_fast_images = []
        
        # Check if the first element is a batch
        if len(standard_images) > 0 and hasattr(standard_images[0], 'shape') and len(standard_images[0].shape) > 3:
            # Images are in batches, flatten them
            for batch in standard_images:
                if isinstance(batch, np.ndarray) and len(batch.shape) > 3:
                    for i in range(batch.shape[0]):
                        flat_std_images.append(batch[i])
                else:
                    flat_std_images.append(batch)
            
            for batch in fast_images:
                if isinstance(batch, np.ndarray) and len(batch.shape) > 3:
                    for i in range(batch.shape[0]):
                        flat_fast_images.append(batch[i])
                else:
                    flat_fast_images.append(batch)
        else:
            # Images are already individual
            flat_std_images = standard_images
            flat_fast_images = fast_images
        
        min_count = min(len(flat_std_images), len(flat_fast_images))
        print(f"Comparing {min_count} images")
      
        # Make sure we compare only the common number of images
                # Make sure we compare only the common number of images
        similarities = []
        for i in range(min_count):
            # Convert to grayscale for SSIM comparison if needed
            std_img = flat_std_images[i]
            fast_img = flat_fast_images[i]
            
            # Debug info and convert to proper arrays if needed
            print(f"Image {i} - normal processing function shape: {std_img.shape if hasattr(std_img, 'shape') else 'unknown'}")
            print(f"Image {i} - fast_img shape: {fast_img.shape if hasattr(fast_img, 'shape') else 'unknown'}")
            
            # Skip images with unknown shapes or complex structures
            if not hasattr(std_img, 'shape') or not hasattr(fast_img, 'shape'):
                print(f"Skipping image {i} due to unknown shape")
                continue
            
            # Convert to proper numpy arrays if they're not already
            try:
                if not isinstance(std_img, np.ndarray):
                    std_img = np.array(std_img)
                if not isinstance(fast_img, np.ndarray):
                    fast_img = np.array(fast_img)
            except ValueError as e:
                print(f"Error converting image {i} to numpy array: {e}")
                print(f"std_img type: {type(std_img)}, fast_img type: {type(fast_img)}")
                if isinstance(std_img, (list, tuple)):
                    print(f"std_img first element type: {type(std_img[0]) if len(std_img) > 0 else 'empty'}")
                if isinstance(fast_img, (list, tuple)):
                    print(f"fast_img first element type: {type(fast_img[0]) if len(fast_img) > 0 else 'empty'}")
                continue
                
            # Make sure both images have same dimensions
            if std_img.shape != fast_img.shape:
                from skimage.transform import resize
                fast_img = resize(fast_img, std_img.shape, anti_aliasing=True)
            
            # Convert images to the same data type before stacking
            std_img = std_img.astype(np.float32)
            fast_img = fast_img.astype(np.float32)
            
            # Create a side-by-side comparison image
            try:
                comparison_img = np.hstack((std_img, fast_img))
                # Convert to uint8 format which OpenCV can handle
                comparison_img = (comparison_img * 255).astype(np.uint8)
                
                # Save the comparison image
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                comparison_path = os.path.join(comparison_dir, f"{file_name}_comparison_{i}.jpg")
                cv2.imwrite(comparison_path, comparison_img)
                
                # Calculate SSIM between images
                similarity = ssim(std_img, fast_img, win_size=3, channel_axis=-1, data_range=1.0)
                similarities.append(similarity)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue
        
        results['avg_similarity'] = np.mean(similarities)
        results['min_similarity'] = np.min(similarities)
        results['max_similarity'] = np.max(similarities)
    
    # Generate summary
    results['summary'] = (
        f"Standard function: {results['standard_time']:.2f}s, {results['standard_image_count']} images\n"
        f"Fast function: {results['fast_time']:.2f}s, {results['fast_image_count']} images\n"
        f"Speedup: {results['speedup_factor']:.2f}x faster\n"
    )
    
    if 'avg_similarity' in results:
        results['summary'] += f"Image similarity: {results['avg_similarity']:.2%} average"

    print(results['summary'])
    return results


#Compare the processing function 

file_path = "/home/emanuelli/Téléchargements/68028001.wav"


# compare_processing_functions(file_path, batch_size=50, start_time=0, end_time=None, wlen=2048,
#                        nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,
#                        target_height_px=677, saving_folder="./compare processing") 













def process_audio_file_old(file_path, saving_folder="./images", batch_size=50, start_time=0, end_time=None, save=False, wlen=2048,

                       nfft=2048, sliding_w=0.4, cut_low_frequency=3, cut_high_frequency=20, target_width_px=903,

                       target_height_px=677):

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

        f, t, Sxx = spectrogram(x_w, fs, nperseg=wlen, noverlap=hop, nfft=nfft, window=win)

        Sxx = 20 * np.log10(np.abs(Sxx))  # Convert to dB



        # Create the spectrogram plot

        fig, ax = plt.subplots()

        ax.pcolormesh(t, f / 1000, Sxx, cmap='gray')

        ax.set_ylim(cut_low_frequency, cut_high_frequency)



        ax.set_axis_off()  # Turn off axis

        fig.set_size_inches(target_width_px / plt.rcParams['figure.dpi'], target_height_px / plt.rcParams['figure.dpi'])



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




        low += samples_per_slice





    plt.close('all')  # Close all figures to release memory



    return images