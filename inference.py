import os
from audioldm import text_to_audio, style_transfer, build_model, save_wave, get_time, round_up_duration, get_duration
import argparse
from inversion import ddim_inversion, ddim_sample
from utils import set_reproducability, load_audio
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
from io import BytesIO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audiosr = build_sr_model(model_name="basic", device=device)
model = build_model(model_name="audioldm-m-full").to(device)

available_sources = ["bass guitar", "violin", "drums",
                        "flute", "piano", "harmonica",
                        "accordion", "trumpet", "xylophone"]

def save_spec(wav=None, sr=None, audio_path=None, name="mel_spectrogram.png"):
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

    # Save the plot to a file
    plt.savefig(name, format='png', dpi=300)
    plt.close()

def edit(
    input_audio,
    target_path,
    edit_mode="replace",
    add_src="None",
    remove_src="None",
    style_prompt="A Jazz style music loop",
    steps=100,
    cfg_scale_src=3,
    cfg_scale_tar=12,
    mixed_sampling=True,
    T_start="far_start",
    close_start=45,
    far_start=65,
    mix_start=30,
    mix_end=32,
    randomize_seed=False,
    scale=1.,
    ):
    steps = int(steps)
    ldm_stable = model

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    x0 = load_audio(input_audio, device=device)

    save_spec(audio_path=input_audio, name=os.path.join(target_path, "input_spec.png"))

    ddim_sampler = DDIMSampler(ldm_stable)
    duration = audioldm.utils.get_duration(input_audio)
    ddim_sampler.make_schedule(int(steps), verbose=False)

    init_latent = ldm_stable.get_first_stage_encoding(
    ldm_stable.encode_first_stage(x0)
    )

    wT, src_latents = ddim_inversion(
            ldm_stable, 
            ddim_sampler, 
            init_latent, 
            input_audio, 
            duration, 
            prompts="", 
            cfg_scale=cfg_scale_src,
            num_inference_steps=steps, 
            skip=steps - far_start, 
            batch_size=1)

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
            single_phase=not mixed_sampling,
            T_start=T_start,
            return_mel=False,
            scale=scale
    )
    
    sf.write(os.path.join(target_path, f"output_audio_{scale}.wav"), waveform[0, 0], samplerate=16000)
    
    waveform = super_resolution(
        audiosr,
        os.path.join(target_path,f"output_audio_{scale}.wav"),
        guidance_scale=3.5,
        ddim_steps=50
    )

    sf.write(os.path.join(target_path, f"output_audio_{scale}.wav"), waveform[0, 0], samplerate=48000)

    save_spec(
        audio_path=os.path.join(target_path,f"output_audio_{scale}.wav"), 
        name=os.path.join(target_path, "output_spec.png")
    )

    print(f"Edited audio saved at: {os.path.join(target_path, f'output_audio_{scale}.wav')}")
    print(f"Input spectrogram saved at: {os.path.join(target_path, 'input_spec.png')}")
    print(f"Output spectrogram saved at: {os.path.join(target_path, 'output_spec.png')}")

    return {
        "edited_audio_path": os.path.join(target_path,f"output_audio_{scale}.wav"),
        "input_spec_path": os.path.join(target_path, "input_spec.png"),
        "output_spec_path": os.path.join(target_path, "output_spec.png")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Editing Script")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--target_path", type=str, required=True, help="Path to save the output files")
    parser.add_argument("--edit_mode", type=str, choices=["addition", "delete", "replace", "style_transfer"], default="addition", help="Edit mode (addition, delete, replace, style_transfer)")
    parser.add_argument("--add_src", type=str, choices=available_sources, default="None", help="Source to add")
    parser.add_argument("--remove_src", type=str, choices=available_sources, default="None", help="Source to remove")
    parser.add_argument("--style_prompt", type=str, default="A Jazz style music loop", help="Style prompt for editing")
    parser.add_argument("--steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--cfg_scale_src", type=float, default=3, help="Source guidance scale")
    parser.add_argument("--cfg_scale_tar", type=float, default=12, help="Target guidance scale")
    parser.add_argument("--mixed_sampling", type=bool, default=True, help="Enable mixed sampling")
    parser.add_argument("--T_start", type=str, default="far_start", help="T-start value (close_start or far_start)")
    parser.add_argument("--close_start", type=int, default=45, help="Close-start percentage")
    parser.add_argument("--far_start", type=int, default=65, help="Far-start percentage")
    parser.add_argument("--mix_start", type=int, default=30, help="Mixing-start percentage")
    parser.add_argument("--mix_end", type=int, default=32, help="Mixing-end percentage")
    parser.add_argument("--randomize_seed", type=bool, default=False, help="Randomize seed")
    parser.add_argument("--scale", type=float, default=1., help="Editing scale")

    args = parser.parse_args()

    result = edit(
        input_audio=args.input_audio,
        target_path=args.target_path,
        edit_mode=args.edit_mode,
        add_src=args.add_src,
        remove_src=args.remove_src,
        style_prompt=args.style_prompt,
        steps=args.steps,
        cfg_scale_src=args.cfg_scale_src,
        cfg_scale_tar=args.cfg_scale_tar,
        mixed_sampling=args.mixed_sampling,
        T_start=args.T_start,
        close_start=args.close_start,
        far_start=args.far_start,
        mix_start=args.mix_start,
        mix_end=args.mix_end,
        randomize_seed=args.randomize_seed,
        scale=args.scale
    )

    print("Editing completed. Results:")
