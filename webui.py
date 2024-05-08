from core import generate
import gradio as gr

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
