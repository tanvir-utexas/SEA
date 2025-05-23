import os
from audioldm import text_to_audio, style_transfer, build_model, save_wave, get_time, round_up_duration, get_duration
import argparse
from inversion import ddim_inversion, ddim_sample
from utils import load_audio
import torch
from torch import inference_mode
from audioldm.latent_diffusion.ddim import DDIMSampler
import torchaudio
from audiosr import super_resolution
from audiosr import build_model as build_sr_model
import gradio as gr
import spaces
from typing import Optional
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import audioldm 
import soundfile as sf
import shutil
import json 
import glob


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the super resolution models
audiosr = build_sr_model(model_name="basic", device=device)

# load the audio editing models
model = build_model(model_name="audioldm-m-full").to(device)


def display_spec(wav=None, sr=None, audio_path=None, name="mel_spectrogram.png"):

    if audio_path is not None:
      y, sr = librosa.load(audio_path, sr=None)
    else:
        y, sr = wav, sr
    
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()

    return name

def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed

@spaces.GPU(duration=200)
def edit(
        input_audio,
        do_inversion,
        wT_holder, 
        src_latents_holder,
        edit_mode,
        add_src,
        remove_src,
        style_prompt,
        steps,
        cfg_scale_src,
        cfg_scale_tar,
        mixed_sampling,
        T_start,
        close_start,
        far_start,
        mix_start,
        mix_end,
        randomize_seed
    ):
    steps = int(steps)
    ldm_stable = model

    temp_path = "./output/temp"
    os.makedirs(temp_path, exist_ok=True)

    if input_audio is None:
        raise gr.Error('Input audio missing!')

    x0 = load_audio(input_audio, device=device)

    mel_input = display_spec(audio_path=input_audio, name=os.path.join(temp_path, "input_spec.png"))

    if wT_holder is None or src_latents_holder is None:
        do_inversion = True

    ddim_sampler = DDIMSampler(ldm_stable)
    duration = audioldm.utils.get_duration(input_audio)
    ddim_sampler.make_schedule(int(steps), verbose=False)

    if do_inversion or randomize_seed:  # always re-run inversion
        # perform the inversion here
        init_latent = ldm_stable.get_first_stage_encoding(
            ldm_stable.encode_first_stage(x0)
        )

        wT, src_latents = ddim_inversion(
                            ldm_stable, 
                            ddim_sampler, 
                            init_latent, 
                            input_audio, 
                            duration, 
                            prompts = "", 
                            cfg_scale=cfg_scale_src,
                            num_inference_steps=steps, 
                            skip= steps - far_start, 
                            batch_size=1)

        wT_holder = wT
        src_latents_holder = src_latents
        do_inversion = False
    else:
        wT = wT_holder.to(device)
        src_latents = [x.to(device) for x in src_latents_holder]

    base_path_audios = "./exemplar_audios"
    src_path = os.path.join(base_path_audios, f"{remove_src}")
    trg_path = os.path.join(base_path_audios, f"{add_src}")

    waveform = ddim_sample(
                        ldm_stable, 
                        ddim_sampler, 
                        src_latents, 
                        input_audio, 
                        duration, 
                        src_prompts="",
                        prompts=style_prompt, 
                        cfg_scale=cfg_scale_tar, 
                        num_inference_steps=steps, 
                        batch_size=1, 
                        source_sample_path=src_path, 
                        target_sample_path=trg_path, 
                        edit_mode=edit_mode,
                        close_start=close_start, 
                        far_start=far_start, 
                        mix_range=[mix_start, mix_end],
                        single_phase= not mixed_sampling,
                        T_start=T_start,
                        return_mel=False)
    
    sf.write(os.path.join(temp_path, "edited_audio.wav"), waveform[0, 0], samplerate=16000)
    
    waveform = super_resolution(
        audiosr,
        os.path.join(temp_path, "edited_audio.wav"),
        guidance_scale=3.5,
        ddim_steps=50
    )

    sf.write(os.path.join(temp_path, "super_res_audio.wav"), waveform[0, 0], samplerate=48000)

    output = (48000, waveform[0][0])
    mel_output = display_spec(
                    audio_path=os.path.join(temp_path, "super_res_audio.wav"), 
                    name= os.path.join(temp_path, "output_spec.png")
                )

    src_latents_holder = [x.cpu() for x in src_latents_holder]

    save_file_paths = [input_audio,
        os.path.join(temp_path, "super_res_audio.wav"), 
        os.path.join(temp_path, "input_spec.png"), 
        os.path.join(temp_path, "output_spec.png")]

    return output, mel_input, mel_output, wT_holder.cpu(), src_latents_holder, do_inversion, save_file_paths


def get_examples_from_dir(base_dir="./Examples"):
    """
    Scans the Examples directory and returns a list of example inputs for gr.Examples.
    Each example includes input_audio, do_inversion, wT_holder, src_latents_holder, edit_mode,
    add_src, remove_src, style_prompt, steps, cfg_scale_src, cfg_scale_tar, mixed_sampling,
    T_start, close_start, far_start, mix_start, mix_end, randomize_seed, output_audio,
    input_spec, output_spec.
    """
    examples = []
    for root, dirs, files in os.walk(base_dir):
        if "hyper_params.json" in files:
            json_path = os.path.join(root, "hyper_params.json")
            with open(json_path, "r") as f:
                params = json.load(f)
            # Find audio and spec files
            input_audio = os.path.join(root, "input_audio.wav")
            output_audio = os.path.join(root, "output_audio.wav")
            input_spec = os.path.join(root, "input_spec.png")
            output_spec = os.path.join(root, "output_spec.png")
            # Fill in the order of inputs for gr.Examples
            example = [
                input_audio,                # input_audio
                params.get("edit_mode", "replace"),
                params.get("add_src", "None"),
                params.get("remove_src", "None"),
                params.get("style_prompt", ""),
                params.get("steps", 100),
                params.get("cfg_scale_src", 3),
                params.get("cfg_scale_trg", 12),
                params.get("mixed_sampling", True),
                "far_start",               # T_start (default, adjust if needed)
                params.get("close_start", 45),
                params.get("far_start", 65),
                params.get("mix_start", 30),
                params.get("mix_end", 32),
                params.get("randomize_seed", False),
                output_audio,              # output_audio
                input_spec,                # input_spectrogram
                output_spec                # output_spectrogram
            ]
            examples.append(example)
    return examples


def save_all(
            save_file_paths,
            folder_name,
            source_prompt,
            source_name,
            edit_mode,
            add_src,
            remove_src,
            style_prompt,
            steps,
            cfg_scale_src,
            cfg_scale_tar,
            mixed_sampling,
            close_start,
            far_start,
            mix_start,
            mix_end,
            randomize_seed            

):
    input_audio_path, output_audio_path = save_file_paths[0], save_file_paths[1] 
    input_spec_path, output_spec_path = save_file_paths[2], save_file_paths[3]

    base_path = "./output/results/"
    path = os.path.join(base_path, source_name, folder_name)
    os.makedirs(path, exist_ok=True)

    # save_waveforms
    shutil.copy(input_audio_path, os.path.join(path, 'input_audio.wav'))
    shutil.copy(output_audio_path, os.path.join(path, 'output_audio.wav'))

    # save specs
    shutil.copy(input_spec_path, os.path.join(path, 'input_spec.png'))
    shutil.copy(output_spec_path, os.path.join(path, 'output_spec.png'))

    # Save dictionary to a JSON file
    data = {
            'source_prompt': source_prompt,
            'edit_mode': edit_mode,
            'add_src': add_src,
            'remove_src': remove_src,
            'style_prompt': style_prompt,
            'steps': steps,
            'cfg_scale_src': cfg_scale_src,
            'cfg_scale_trg': cfg_scale_tar,
            'mixed_sampling': mixed_sampling,
            'close_start': close_start,
            'far_start': far_start,
            'mix_start': mix_start,
            'mix_end': mix_end,
            'randomize_seed': randomize_seed            
    }

    with open(os.path.join(path, 'hyper_params.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Saved all files!")

intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">Semantic Audio Editing System</h1>
<h2 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">Advanced Audio Manipulation through Semantic Embedding Arithmetic and Diffusion Models</h2>
"""

help = """
<div style="font-size:medium">
<b>Instructions for Audio Editing:</b><br>
<ul style="line-height: normal">
<li>For REPLACEMENT mode:
    <ul>
        <li>Set edit mode to 'replace'</li>
        <li>Select both 'drop source' (what to remove) and 'add source' (what to add)</li>  
    </ul>
</li>
<li>For ADDITION mode:
    <ul>
        <li>Set edit mode to 'addition'</li>
        <li>Select only 'add source' (source to add)</li>
        <li>'Drop source' can be left as None</li>
    </ul>
</li>
<li>For DROP mode:
    <ul>
        <li>Set edit mode to 'delete'</li>
        <li>Select only 'drop source' (source to remove)</li>
        <li>'Add source' can be left as None</li>
    </ul>
</li>
<li>For STYLE TRANSFER mode:
    <ul>
        <li>Set edit mode to 'style_transfer'</li>
        <li>Enter desired style in the 'Style Prompt' field</li>
        <li>Sources can be left as None</li>
    </ul>
</li>
<li>Note: Input audio is limited to 10 seconds duration</li>
</ul>
</div>
"""

with gr.Blocks(css='style.css') as demo:  
    def reset_do_inversion(do_inversion_user, do_inversion):
        do_inversion = True
        do_inversion_user = True
        return do_inversion_user, do_inversion

    # handle the case where the user clicked the button but the inversion was not done
    def clear_do_inversion_user(do_inversion_user):
        do_inversion_user = False
        return do_inversion_user

    def post_match_do_inversion(do_inversion_user, do_inversion):
        if do_inversion_user:
            do_inversion = True
            do_inversion_user = False
        return do_inversion_user, do_inversion

    gr.HTML(intro)
    wT_holder = gr.State()
    src_latents_holder = gr.State()
    save_file_paths = gr.State()
    do_inversion = gr.State(value=True)  # To save some runtime when editing the same thing over and over
    do_inversion_user = gr.State(value=False)

    with gr.Group():
        gr.Markdown("💡 **note**: Only 10s duration input is supported")
        with gr.Row():
            with gr.Row():
                input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", editable=True, label="Input Audio",
                                    interactive=True, scale=1)
                output_audio = gr.Audio(label="Edited Audio", interactive=False, scale=1)
    
    with gr.Accordion("Show Spectrogram", open=False):
        with gr.Row():
            input_spectrogram = gr.Image(label="Input Spec", type="filepath")
            output_spectrogram = gr.Image(label="Edited Spec", type="filepath")

    with gr.Row():
        # edit mode
        edit_mode = gr.Dropdown(label="Edit Mode",
                               choices=["addition",
                                        "delete",
                                        "replace",
                                        "style_transfer"],
                               info="Choose a mode for editing",
                               value="replace", interactive=True, type="value", scale=2)
        
        # Remove source (10 sources)
        remove_src = gr.Dropdown(label="Drop Source",
                               choices=["acoustic guitar",
                                        "bass guitar",
                                        "violin",
                                        "drums",
                                        "flute",
                                        "piano",
                                        "harmonica",
                                        "accordion",
                                        "trumpet",
                                        "xylophone",
                                        "None"],
                               info="Choose a source to be removed",
                               value="None", interactive=True, type="value", scale=2)

        add_src = gr.Dropdown(label="Add Source",
                               choices=["acoustic guitar",
                                        "bass guitar",
                                        "violin",
                                        "drums",
                                        "flute",
                                        "piano",
                                        "harmonica",
                                        "accordion",
                                        "trumpet",
                                        "xylophone",
                                        "None"],
                               info="Choose a source to be added",
                               value="None", interactive=True, type="value", scale=2)

    with gr.Row():
        style_prompt = gr.Textbox(label="Style Prompt", info="Suffix for style transfer",
                        placeholder="A Jazz style music loop",
                        lines=1, interactive=True)
    with gr.Row():
        submit = gr.Button("Edit")
        save = gr.Button("Save All")

    with gr.Accordion("Sampling Options", open=False):
        with gr.Row():
            mixed_sampling = gr.Checkbox(label='Mixed Sampling', value=True)

            T_start = gr.Dropdown(label="T-start",
                               choices=["close_start",
                                        "far_start"],
                               info="For single sampling, choose the T-start value",
                               value="far_start", interactive=True, type="value")

            close_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label="Close-start (%)", interactive=True, scale=3,
                                info="Sampling closer to original audio. Lower T-start -> closer to original audio.")

            far_start = gr.Slider(minimum=15, maximum=85, value=65, step=1, label="Far-start (%)", interactive=True, scale=3,
                                info="Sampling closer to stronger edit. Higher T-start -> stronger edit.")

        with gr.Row():
            mix_start = gr.Slider(minimum=15, maximum=85, value=30, step=1, label="Mixing-start (%)", interactive=True, scale=3,
                                info="Where mixing starts.")

            mix_end = gr.Slider(minimum=15, maximum=85, value=32, step=1, label="Mixing-end (%)", interactive=True, scale=3,
                                info="Where mixing ends.")

    with gr.Accordion("More Options", open=False):
        with gr.Row():
            cfg_scale_src = gr.Number(value=3, minimum=0.5, maximum=25, precision=None,
                                      label="Source Guidance Scale", interactive=True, scale=1)
            cfg_scale_tar = gr.Number(value=12, minimum=0.5, maximum=25, precision=None,
                                      label="Target Guidance Scale", interactive=True, scale=1)
            steps = gr.Number(value=100, step=1, minimum=20, maximum=300,
                              info="Higher values (e.g. 200) yield higher-quality generation.",
                              label="Num Diffusion Steps", interactive=True, scale=1)
        with gr.Row():
            source_prompt = gr.Textbox(label="Source Prompt", lines=2, interactive=True,
                                    info="Describe the original audio input. Used for saving.",
                                    placeholder="A music loop with piano and drums",)
            source_name = gr.Textbox(label="Source file name", lines=2, interactive=True,
                                    info="Provide the source file name. Used for saving.",
                                    placeholder="Piano and Drums (Sample 3)",)

        with gr.Row():
            seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
            randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
            folder_name = gr.Textbox(label="Folder Name", placeholder="Sample dir name")
            length = gr.Number(label="Length", interactive=False, visible=False)

    with gr.Accordion("Help💡", open=False):
        gr.HTML(help)

    submit.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=[seed], queue=False).then(
            fn=clear_do_inversion_user, inputs=[do_inversion_user], outputs=[do_inversion_user]).then(
           fn=edit,
           inputs=[
                    input_audio,
                    do_inversion,
                    wT_holder, 
                    src_latents_holder,
                    edit_mode,
                    add_src,
                    remove_src,
                    style_prompt,
                    steps,
                    cfg_scale_src,
                    cfg_scale_tar,
                    mixed_sampling,
                    T_start,
                    close_start,
                    far_start,
                    mix_start,
                    mix_end,
                    randomize_seed
                   ],
           outputs=[output_audio, 
                    input_spectrogram,
                    output_spectrogram,
                    wT_holder, 
                    src_latents_holder, 
                    do_inversion,
                    save_file_paths]
        ).then(post_match_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion]
               )
               
               
    save.click(
        save_all, 
        inputs=[
            save_file_paths,
            folder_name,
            source_prompt,
            source_name,
            edit_mode,
            add_src,
            remove_src,
            style_prompt,
            steps,
            cfg_scale_src,
            cfg_scale_tar,
            mixed_sampling,
            close_start,
            far_start,
            mix_start,
            mix_end,
            randomize_seed            
        ])
    
    # If sources changed we have to rerun inversion
    input_audio.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    edit_mode.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    remove_src.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    add_src.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    style_prompt.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    far_start.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])

    cfg_scale_src.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])
    cfg_scale_tar.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion]) 
    steps.change(fn=reset_do_inversion, inputs=[do_inversion_user, do_inversion], outputs=[do_inversion_user, do_inversion])

    gr.Examples(
        label="Examples",
        examples=get_examples_from_dir("./Examples"),
        inputs=[
            input_audio,
            edit_mode,
            add_src,
            remove_src,
            style_prompt,
            steps,
            cfg_scale_src,
            cfg_scale_tar,
            mixed_sampling,
            T_start,
            close_start,
            far_start,
            mix_start,
            mix_end,
            randomize_seed,
            output_audio,
            input_spectrogram,
            output_spectrogram
        ],
        outputs=[output_audio, input_spectrogram, output_spectrogram]
    )

    demo.queue()
    demo.launch()
