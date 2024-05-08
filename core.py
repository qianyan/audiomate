import datetime
import hashlib
import io
import json
import os
from typing import Optional, TypedDict

import gradio as gr
import matplotlib
import matplotlib.figure as mpl_fig
import numpy as np
import scipy
from matplotlib import pyplot as plt
from transformers import AutoProcessor, MusicgenForConditionalGeneration

sep = ","

matplotlib.use("agg")


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


def gen(model, params):
    text_array = [text.strip() for text in params["text"].split(sep)]
    model_name = params["model"]
    processor = AutoProcessor.from_pretrained(model_name)

    print(text_array)
    inputs = processor(
        # text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        text=text_array,
        padding=True,
        return_tensors="pt",
    )

    # 256 tokens -> 5s long music, so (256/5)*duration will generate the long music
    return model.generate(**inputs, max_new_tokens=(256 // 5) * params["duration"])


def save_generation(
        audio_array: np.ndarray,
        sample_rate: int,
        params: MusicGenGeneration
):
    prompt = params["text"]
    date = get_date_string()
    title = prompt[:20].replace(" ", "_")
    base_filename = create_base_filename(title, "outputs", model="musicgen", date=date)

    filename, filename_png, filename_json, filename_npz = get_filenames(base_filename)
    stereo = audio_array.shape[0] == 2
    if stereo:
        print("stere here.")
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


def generate(params: MusicGenGeneration):
    model_name = params["model"]
    text = params["text"]

    model = load_model(model_name)
    output = gen(model, params)[0, 0].numpy()

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


def get_filenames(base_filename: str):
    filename = f"{base_filename}.wav"
    filename_png = f"{base_filename}.png"
    filename_json = f"{base_filename}.json"
    filename_npz = f"{base_filename}.npz"
    return filename, filename_png, filename_json, filename_npz


def get_date_string():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def plot_waveform(audio_array: np.ndarray):
    fig = plt.figure(figsize=(10, 3))
    plt.style.use("dark_background")
    plt.plot(audio_array, color="orange")
    plt.axis("off")
    return fig


def figure_to_image(fig: mpl_fig.Figure):
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return data.reshape((int(h), int(w), -1))


def save_waveform_plot(audio_array: np.ndarray, filename_png: str):
    fig = plot_waveform(audio_array)
    plt.savefig(filename_png)
    plt.close()
    return figure_to_image(fig)


def audio_array_to_sha256(audio_array: np.ndarray) -> str:
    return hashlib.sha256(audio_array.tobytes()).hexdigest()


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


def replace_path_sep(title: Optional[str]) -> str:
    return "None" if title is None else title.replace(os.path.sep, "_")

def generation_tab_musicgen():
    with gr.Tab("MusicGen") as tab:
        musicgen_atom = gr.JSON(
            visible=False,
            value={
                "text": "",
                "melody": None,
                "model": "melody",
                "duration": 10,
                "topk": 250,
                "topp": 0,
                "temperature": 1.0,
                "cfg_coef": 3.0,
                "seed": -1,
                "use_multi_band_diffusion": False,
            },
        )
        with gr.Row(equal_height=False):
            with gr.Column():
                text = gr.Textbox(
                    label="Prompt", lines=3, placeholder="Enter text here..."
                )
                model = gr.Radio(
                    [
                        "facebook/musicgen-melody",
                        "facebook/musicgen-medium",
                        "facebook/musicgen-small",
                        "facebook/musicgen-large",
                        "facebook/audiogen-medium",
                        "facebook/musicgen-melody-large",
                        "facebook/musicgen-stereo-small",
                        "facebook/musicgen-stereo-medium",
                        "facebook/musicgen-stereo-melody",
                        "facebook/musicgen-stereo-large",
                        "facebook/musicgen-stereo-melody-large",
                    ],
                    label="Model",
                    value="facebook/musicgen-small",
                )
                submit = gr.Button("Generate", variant="primary")
                with gr.Column():
                    duration = gr.Slider(
                        minimum=1,
                        maximum=360,
                        value=10,
                        label="Duration",
                    )
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=0.0,
                        label="Top-p",
                        interactive=True,
                        step=0.05,
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        value=1.0,
                        label="Temperature",
                        interactive=True,
                        step=0.05,
                    )
                    cfg_coef = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        value=3.0,
                        label="Classifier Free Guidance",
                        interactive=True,
                        step=0.1,
                    )
                use_multi_band_diffusion = gr.Checkbox(
                    label="Use Multi-Band Diffusion (High VRAM Usage)",
                    value=False,
                )
        with gr.Column():
            output = gr.Audio(
                label="Generated Music",
                type="numpy",
                interactive=False,
                elem_classes="tts-audio",
            )
            image = gr.Image(label="Waveform", shape=(None, 100), elem_classes="tts-image")
    inputs = [
        text,
        model,
        duration,
        topk,
        topp,
        temperature,
        cfg_coef,
        use_multi_band_diffusion,
    ]

    def update_components(x):
        return {
            text: x["text"],
            model: x["model"],
            duration: x["duration"],
            topk: x["topk"],
            topp: x["topp"],
            temperature: x["temperature"],
            cfg_coef: x["cfg_coef"],
            use_multi_band_diffusion: x["use_multi_band_diffusion"],
        }

    musicgen_atom.change(
        fn=update_components,
        inputs=musicgen_atom,
        outputs=inputs,
    )

    def update_json(
            text,
            model,
            duration,
            topk,
            topp,
            temperature,
            cfg_coef,
            use_multi_band_diffusion,
    ):
        return {
            "text": text,
            "model": model,
            "duration": int(duration),
            "topk": int(topk),
            "topp": float(topp),
            "temperature": float(temperature),
            "cfg_coef": float(cfg_coef),
            "use_multi_band_diffusion": bool(use_multi_band_diffusion),
        }

    result_json = gr.JSON(
        visible=False,
    )

    submit.click(
        fn=update_json,
        inputs=inputs,
        outputs=[musicgen_atom],
    ).then(
        fn=generate,
        inputs=[musicgen_atom],
        outputs=[output, image, result_json],
        api_name="musicgen",
    )

    return tab, musicgen_atom


if __name__ == "__main__":
    with gr.Blocks() as demo:
        generation_tab_musicgen()
    demo.queue().launch()
