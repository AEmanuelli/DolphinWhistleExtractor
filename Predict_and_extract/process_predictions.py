"""Post-processing of predictions: extract the detected whistles as audio/video clips.

Reads the ``*.wav_predictions.csv`` files written by
``predict_and_extract_online.process_predict_extract``, merges neighbouring
detections into intervals and cuts the corresponding segments out of the
original recordings (audio only, or video with/without the hydrophone audio).

Audio clips are cut with ``soundfile``: samples are copied as they are, so the
original sampling rate (e.g. 96 kHz) and bit depth are preserved.
Video extraction needs ``moviepy`` (imported lazily).
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:  # works both as a script directory import and as a package import
    from utils import (fusionner_intervalles, fusionner_intervalles_avec_seuil, lire_csv_extraits,
                       transform_file_name, trouver_fichier_video)
except ImportError:  # pragma: no cover
    from .utils import (fusionner_intervalles, fusionner_intervalles_avec_seuil, lire_csv_extraits,
                        transform_file_name, trouver_fichier_video)

logger = logging.getLogger(__name__)

NO_EXTRACT_FOLDER = "pas_d_extraits"


def audioextraits(intervalles, fichier_audio, dossier_sortie_audio):
    """
    Cut ``intervalles`` [(start_s, end_s), ...] out of an audio file.

    Clips are written to ``<dossier_sortie_audio>/<recording name>/extrait_<start>_<end>.wav``
    with the sampling rate and sample format of the source. Existing clips are kept.

    Returns:
        list of the clip paths (existing or newly written).
    """
    import soundfile as sf

    filename = os.path.splitext(os.path.basename(fichier_audio))[0]
    dossier_du_fichier = os.path.join(dossier_sortie_audio, filename)
    os.makedirs(dossier_du_fichier, exist_ok=True)

    chemins = []
    with sf.SoundFile(fichier_audio) as audio:
        subtype = audio.subtype if sf.check_format('WAV', audio.subtype) else 'PCM_16'
        dtype = 'float32' if subtype in ('FLOAT', 'DOUBLE') else 'int32'
        for debut, fin in tqdm(intervalles, desc=f'Extraction pour {filename}', leave=False):
            chemin_sortie = os.path.join(dossier_du_fichier, f'extrait_{debut}_{fin}.wav')
            chemins.append(chemin_sortie)
            if os.path.exists(chemin_sortie):
                continue
            start = max(0, int(debut * audio.samplerate))
            stop = min(audio.frames, int(fin * audio.samplerate))
            if stop <= start:
                chemins.pop()
                continue
            audio.seek(start)
            sf.write(chemin_sortie, audio.read(stop - start, dtype=dtype), audio.samplerate, subtype=subtype)
    return chemins


def extraire_extraits_video(intervalles, fichier_video, dossier_sortie_video):
    """Cut ``intervalles`` out of a video file (requires moviepy)."""
    from moviepy.editor import VideoFileClip

    video = VideoFileClip(fichier_video)
    try:
        for debut, fin in tqdm(intervalles, desc=f'Extraction pour {fichier_video}', leave=False):
            chemin_sortie = os.path.join(dossier_sortie_video, f'extrait_{debut}_{fin}.mp4')
            if not os.path.exists(chemin_sortie):
                video.subclip(debut, fin).write_videofile(chemin_sortie, verbose=False)
    finally:
        video.close()


def _write_note(folder_path, note_name, message):
    """Leave a small text file explaining why a recording has no extract."""
    dossier = os.path.join(folder_path, NO_EXTRACT_FOLDER)
    os.makedirs(dossier, exist_ok=True)
    with open(os.path.join(dossier, note_name), 'w') as txt_file:
        txt_file.write(message)
    logger.info(message)


def find_recording(recording_folder_path, folder_name):
    """Path of the recording (.wav or .flac) named ``folder_name``, or None."""
    for extension in (".wav", ".flac", ".WAV", ".FLAC"):
        candidate = os.path.join(recording_folder_path, folder_name + extension)
        if os.path.exists(candidate):
            return candidate
    return None


def process_non_empty_file(prediction_file_path, folder_name, recording_folder_path, exit, folder_path,
                           audio=True, audio_only=True):
    """Extract the clips of one recording from its (non-empty) predictions CSV."""
    intervalles = lire_csv_extraits(prediction_file_path)

    if audio_only:
        intervalles_fusionnes = fusionner_intervalles_avec_seuil(intervalles, duration_threshold=5, fusion_threshold=3)
        fichier_audio = find_recording(recording_folder_path, folder_name)
        if fichier_audio is None:
            _write_note(folder_path, "No_Audio_found.txt", f"No audio file found for {folder_name}")
            return
        os.makedirs(exit, exist_ok=True)
        audioextraits(intervalles_fusionnes, fichier_audio, exit)
        return

    intervalles_fusionnes = fusionner_intervalles(intervalles, hwindow=5)
    fichier_video = trouver_fichier_video(folder_name, recording_folder_path)
    if not fichier_video:
        _write_note(folder_path, "No_Video_found.txt", f"No video found for {folder_name}")
        return

    dossier_sortie_video = os.path.join(folder_path, "extraits")
    if audio:
        from vidéoaudio import vidéoetaudio
        fichier_audio = os.path.join(recording_folder_path, folder_name + ".wav")
        dossier_sortie_video_et_audio = dossier_sortie_video + "_avec_audio"
        os.makedirs(dossier_sortie_video_et_audio, exist_ok=True)
        vidéoetaudio(intervalles_fusionnes, fichier_video, fichier_audio, dossier_sortie_video_et_audio)
    else:
        os.makedirs(dossier_sortie_video, exist_ok=True)
        extraire_extraits_video(intervalles_fusionnes, fichier_video, dossier_sortie_video)


def handle_empty_file(folder_path, folder_name):
    t_file_name = transform_file_name(folder_name)
    _write_note(folder_path, "No_whistles_detected.txt", f"No whistles detected in {t_file_name}")


def handle_missing_file(folder_path, folder_name):
    t_file_name = transform_file_name(folder_name)
    _write_note(folder_path, "No_CSV_found.txt", f"No CSV found in {t_file_name}")


def process_prediction_file(prediction_file_path, folder_name, recording_folder_path, folder_path, exit, audio,
                            audio_only=True):
    """Process one predictions CSV (missing, empty or with detections)."""
    if not os.path.exists(prediction_file_path):
        handle_missing_file(folder_path, folder_name)
        return

    with open(prediction_file_path, 'r') as file:
        has_detections = sum(1 for line in file if line.strip()) > 1  # more than the header

    if has_detections:
        process_non_empty_file(prediction_file_path, folder_name, recording_folder_path, folder_path=folder_path,
                               exit=exit, audio=audio, audio_only=audio_only)
    else:
        handle_empty_file(folder_path, folder_name)


def process_folder(root, folder_name, recording_folder_path, folder_path, exit, audio=True, audio_only=True):
    """Process the predictions CSV of one recording folder, if there is one."""
    prediction_file_path = os.path.join(root, folder_name, folder_name + ".wav_predictions.csv")
    if os.path.exists(prediction_file_path):
        process_prediction_file(prediction_file_path, folder_name, recording_folder_path, folder_path,
                                audio=audio, audio_only=audio_only, exit=exit)


def process_prediction_files_in_folder(root, recording_folder_path, max_workers=8, audio=True, audio_only=True,
                                       exit=None):
    """
    Extract the detected whistles of every recording analysed in ``root``.

    Args:
        root: Folder containing one sub-folder per recording (output of the detection step).
        recording_folder_path: Folder containing the original recordings (and videos).
        max_workers: Number of recordings processed in parallel.
        audio: When extracting video, also mux the hydrophone audio into the clips.
        audio_only: Extract audio clips only (no video needed).
        exit: Folder receiving the audio clips (default: ``<root>/extracted_audio``).
    """
    if exit is None:
        exit = os.path.join(root, "extracted_audio")

    folders = [name for name in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, name)) and os.path.join(root, name) != exit]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_folder, root, name, recording_folder_path, os.path.join(root, name),
                            audio=audio, audio_only=audio_only, exit=exit): name
            for name in folders
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting whistles"):
            try:
                future.result()
            except Exception as e:  # keep going with the other recordings
                logger.error("Extraction failed for %s: %s", futures[future], e)
