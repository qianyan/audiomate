import json
from typing import TypedDict

import numpy as np
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration

from plot import save_waveform_plot
from util import audio_array_to_sha256, create_base_filename, get_date_string, get_filenames


class MusicGenGeneration(TypedDict):
    model: str
    text: str
    duration: int
    topk: int
    topp: float
    temperature: float
    cfg_coef: float
    use_multi_band_diffusion: bool


def load_model(name="facebook/musicgen-small"):
    return MusicgenForConditionalGeneration.from_pretrained(name)


def generate_audio(model, params):
    text_array = [text.strip() for text in params["text"].split(",")]
    model_name = params["model"]
    processor = AutoProcessor.from_pretrained(model_name)

    print(text_array)
    inputs = processor(
        # text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        text=text_array,
        padding=True,
        return_tensors="pt",
    )

    # 256 tokens -> 5s long music, so (256/5)*duration will generate the longer music
    return model.generate(**inputs, max_new_tokens=(256 // 5) * params["duration"])


def save_generation(
        audio_array: np.ndarray,
        sample_rate: int,
        params: MusicGenGeneration
):
    prompt = params["text"]
    date = get_date_string()
    base_filename = the_base_filename(date, prompt)

    filename, filename_png, filename_json = get_filenames(base_filename)

    stereo = audio_array.shape[0] == 2
    if stereo:
        audio_array = np.transpose(audio_array)

    scipy.io.wavfile.write(filename, sample_rate, audio_array)

    plot = save_waveform_plot(audio_array, filename_png)

    metadata = generate_and_save_metadata(
        prompt=prompt,
        date=date,
        filename_json=filename_json,
        params=params,
        audio_array=audio_array,
    )
    return filename, plot, metadata


def the_base_filename(date, prompt):
    title = prompt[:20].replace(" ", "_")
    return create_base_filename(title, "outputs", model="musicgen", date=date)


def generate(params: MusicGenGeneration):
    model = load_model(params["model"])

    output = generate_audio(model, params)[0, 0].numpy()

    sample_rate = model.config.audio_encoder.sampling_rate

    filename, plot, _metadata = save_generation(
        audio_array=output,
        sample_rate=sample_rate,
        params=params
    )

    return [
        (sample_rate, output),
        plot,
        _metadata,
    ]


def generate_and_save_metadata(
        prompt: str,
        date: str,
        filename_json: str,
        params: MusicGenGeneration,
        audio_array: np.ndarray,
):
    metadata = {
        "_version": "0.0.1",
        "_hash_version": "0.0.3",
        "_type": "musicgen",
        "models": {},
        "prompt": prompt,
        "hash": audio_array_to_sha256(audio_array),
        "date": date,
        **params,
    }
    with open(filename_json, "w") as outfile:
        json.dump(metadata, outfile, indent=2)

    return metadata
