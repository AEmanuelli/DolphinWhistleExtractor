import os
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm

def vidéoetaudio(intervalles, fichier_video, fichier_audio, dossier_sortie_video):
    # Load the video and audio
    video = VideoFileClip(fichier_video)
    audio = AudioFileClip(fichier_audio)
    
    # Calculate the total number of extracts to generate
    total_extraits = len(intervalles)
    
    # Display a progress bar
    with tqdm(total=total_extraits, desc=f'Extraction for {fichier_video}') as pbar:
        # Iterate over the intervals
        for i, intervalle in enumerate(intervalles):
            debut, fin = intervalle
            nom_sortie = f'extrait_{debut}_{fin}.mp4'  # Output name based on the interval

            chemin_sortie = os.path.join(dossier_sortie_video, nom_sortie)  # Full output path
            if not os.path.exists(chemin_sortie):
                # Extract the video segment corresponding to the interval
                extrait_video = video.subclip(debut, fin)
                extrait_audio = audio.subclip(debut, fin)
                # Add the audio to the extracted video
                extrait_video = extrait_video.set_audio(extrait_audio)
                # Save the video extract with audio
                extrait_video.write_videofile(chemin_sortie, codec='libx264', audio_codec='aac', fps=video.fps, verbose=False)
            else : 
                print("zbzbz")
            pbar.update(1)
        
    # Free up memory by closing the VideoFileClip and AudioFileClip objects
    video.close()
    audio.close()


# vidéoetaudio([(10, 20)], "/media/DOLPHIN_ALEXIS1/2023/Exp_28_Mar_2024_1445_cam_all.mp4", "/media/DOLPHIN_ALEXIS1/2023/Exp_28_Mar_2024_1445_channel_1.wav", "/users/zfne/emanuell/Desktop/test")