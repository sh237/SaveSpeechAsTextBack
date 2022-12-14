# convert_audio_file.py

from django.core.files import File
from pydub import AudioSegment
from pathlib import Path
import soundfile as sf
import subprocess
import os


def convert_to_wav(audio_file, target_filetype='wav', content_type='audio/x-caf',bitrate="128k"):

    file_path = audio_file.temporary_file_path()
    new_path = file_path[:-3] + target_filetype
    subprocess.run(f"afinfo {file_path}", shell=True)
    subprocess.run(f"afconvert -f WAVE -d LEI16 -c 1 -b 128000 {file_path} {new_path}", shell=True)

    converted_audiofile = File(
                file=open(new_path, 'rb'),
                name=Path(new_path)
            )
    converted_audiofile.name = Path(new_path).name
    converted_audiofile.content_type = content_type
    converted_audiofile.size = os.path.getsize(new_path)
    return converted_audiofile