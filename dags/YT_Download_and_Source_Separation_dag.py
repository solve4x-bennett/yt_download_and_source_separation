import re
import os
import soundfile
import librosa.display

import numpy as np
import moviepy.editor as mp

from pathlib import Path
from yt_dlp import YoutubeDL

from airflow.decorators import dag, task

DATA_DIR = Path("/usr/local/airflow/data")

@dag(
    dag_id="YT_Download_and_Source_Separator",
    schedule=None,
    params={
        "YouTube Link": None,
    }
)
def yt_download_and_source_separator():
    """
    """
    
    @task(task_id="download_youtube_audio", multiple_outputs=True)
    def download_youtube_audio(**context) -> str:
        """
        Downloads youtube video into .mp3 file format and returns the Path to the download location

        Args:
            link (str): Youtube link

        Returns:
            str: File path
        """
        link = context["params"]["YouTube Link"]

        with YoutubeDL({"outtmpl": "%(title)s"}) as video:
            video.download(link)

            info_dict = video.extract_info(link, download=False)
            audio_filename = f"{re.sub('[\W_]+', '_', info_dict['title'])}.mp3"

            file_found = False
            for file in os.listdir():
                    if f"{info_dict['title']}" in file:
                            video_filename = file
                            file_found = True
                    else:
                            continue

            if not file_found:
                    raise FileNotFoundError(f"No file named {info_dict['title']}")

            clip = mp.VideoFileClip(video_filename)

            print("\nConverting file to mp3 ....")

            # create dir in data folder for outputs
            output_path = Path(DATA_DIR / f"{info_dict['title']} Outputs")
            output_path.mkdir(parents=True, exist_ok=True)

            # save audio file
            clip.audio.write_audiofile(filename=output_path / audio_filename, verbose=False, logger=None)

            print("Conversion Complete!")

            os.remove(video_filename)

            clip.close()

        # set output path to str because Path is not JSON serializable
        return {"output_path": str(output_path), "audio_filename": audio_filename}

    @task(task_id="source_separate")
    def source_separation(output_path: str, audio_filename: str) -> None:
        """
        Separates vocals from background music

        Args:
            audio_filename (str): The name of the audio file saved

        Returns:
            None
        """
        output_path = Path(output_path)

        y, sr = librosa.load(output_path / audio_filename)

        # compute magnitude and phase
        S_full, phase = librosa.magphase(librosa.stft(y))

        # Compare frames using cosine similarity, and aggregate similar frames
        # by taking their (pre-frequency) median value

        S_filter = librosa.decompose.nn_filter(
            S_full, 
            aggregate=np.median, 
            metric="cosine", 
            width=int(librosa.time_to_frames(2, sr=sr))
        )

        # Output of the filter shouldn't be greater than the input
        S_filter = np.minimum(S_full, S_filter)

        # Using a margin to reduce bleed between the vocals and instrumentation masks
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(
            S_filter,
            margin_i * (S_full - S_filter),
            power=power
        )

        mask_v = librosa.util.softmask(
            S_full - S_filter,
            margin_v * S_filter,
            power=power
        )

        # once we have the mask, multiply them with the input spectrum to separate components
        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        foreground_vocal = librosa.istft(S_foreground * phase)
        background_music = librosa.istft(S_background * phase)

        soundfile.write(output_path / "foreground_vocals.wav", foreground_vocal, sr)
        soundfile.write(output_path / "background_music.wav", background_music, sr)

    output_info = download_youtube_audio()
    source_separation(output_path=output_info["output_path"], audio_filename=output_info["audio_filename"])

yt_download_and_source_separator()