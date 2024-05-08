import datetime
import hashlib
import os
from typing import Optional

import numpy as np


def _create_base_filename(
        title: Optional[str], output_path: str, model: str, date: str
) -> str:
    base = f"{date}__{model}__{replace_path_sep(title)}"
    return os.path.join(output_path, base, base)


def create_base_filename(
        title: Optional[str], output_path: str, model: str, date: str
) -> str:
    base_filename = _create_base_filename(title, output_path, model, date)

    base_directory = os.path.dirname(base_filename)
    os.makedirs(base_directory, exist_ok=True)

    return base_filename


def get_filenames(base_filename: str):
    filename = f"{base_filename}.wav"
    filename_png = f"{base_filename}.png"
    filename_json = f"{base_filename}.json"
    return filename, filename_png, filename_json


def get_date_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def replace_path_sep(title: Optional[str]) -> str:
    return "None" if title is None else title.replace(os.path.sep, "_")


def audio_array_to_sha256(audio_array: np.ndarray) -> str:
    return hashlib.sha256(audio_array.tobytes()).hexdigest()
