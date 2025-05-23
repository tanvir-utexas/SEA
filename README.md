# Caption-Free Diffusion Audio Editing via Semantic Embedding Arithmetic

## Overview

This repository provides tools and code for **Caption-Free Diffusion Audio Editing** using **Semantic Embedding Arithmetic**. This approach enables flexible audio editing without requiring textual captions, leveraging advanced diffusion models and semantic embeddings.

## Features

- **Gradio Demo:** Interactive web interface for real-time audio editing.
- **Inference Script:** Command-line tool for batch audio editing.
- **Semantic Editing:** Supports addition, deletion, replacement, and style transfer of audio attributes.

---

## Installation

### Using `pip`
Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Using `conda`
Alternatively, create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate sea-audio-edit
```

---

## Usage

### 1. Running the Gradio Demo

Launch the Gradio demo to interactively edit audio files:

```bash
python gradio_demo.py
```

Once the server starts, open the provided URL in your browser to access the interface.

#### Features of the Gradio Demo:
- Upload or record audio files (up to 10 seconds).
- Choose from various editing modes:
  - **Addition:** Add a semantic attribute to the audio.
  - **Delete:** Remove a semantic attribute from the audio.
  - **Replace:** Replace one attribute with another.
  - **Style Transfer:** Apply a specific style to the audio.
- Visualize input and output spectrograms.
- Save edited audio and spectrograms for further use.

---

### 2. Running Inference via Command Line

Use the `inference.py` script to edit audio files in batch mode:

```bash
python inference.py --input_audio <path_to_input_audio> --target_path <path_to_save_outputs> \
    --edit_mode <edit_mode> --add_src <source_to_add> --remove_src <source_to_remove> \
    --style_prompt <style_prompt> --steps <num_steps> --cfg_scale_src <source_guidance_scale> \
    --cfg_scale_tar <target_guidance_scale> --mixed_sampling <true_or_false> \
    --T_start <close_start_or_far_start> --close_start <close_start_percentage> \
    --far_start <far_start_percentage> --mix_start <mix_start_percentage> \
    --mix_end <mix_end_percentage> --randomize_seed <true_or_false>
```

#### Arguments:
- `--input_audio`: Path to the input audio file (e.g., `.wav`).
- `--target_path`: Directory to save the output files.
- `--edit_mode`: Type of editing operation (`addition`, `delete`, `replace`, `style_transfer`).
- `--add_src`: Source to add (e.g., `flute`, `piano`, etc.).
- `--remove_src`: Source to remove (e.g., `drums`, `bass guitar`, etc.).
- `--style_prompt`: Style description for style transfer (e.g., "A happy upbeat music loop").
- `--steps`: Number of diffusion steps (higher values yield better quality).
- `--cfg_scale_src`: Guidance scale for the source.
- `--cfg_scale_tar`: Guidance scale for the target.
- `--mixed_sampling`: Enable or disable mixed sampling (`true` or `false`).
- `--T_start`: Starting point for sampling (`close_start` or `far_start`).
- `--close_start`: Percentage for close-start sampling.
- `--far_start`: Percentage for far-start sampling.
- `--mix_start`: Percentage where mixing starts.
- `--mix_end`: Percentage where mixing ends.
- `--randomize_seed`: Randomize the seed for reproducibility (`true` or `false`).

#### Example Command:
```bash
python inference.py --input_audio samples/input.wav --target_path results/ \
    --edit_mode replace --add_src flute --remove_src drums \
    --steps 100 --cfg_scale_src 3 --cfg_scale_tar 12 --mixed_sampling true \
    --T_start far_start --close_start 40 --far_start 60 --mix_start 30 --mix_end 32
```

This command replaces "drums" with "flute" in `input.wav` and saves the results in the `results/` directory.

---

### 3. Reproducing Results with Predefined Examples

The repository includes predefined examples in the `Examples` directory. Each example contains:
- `hyper_params.json`: Configuration file with editing parameters.
- `input_audio.wav`: Input audio file.
- `input_spec.png`: Input spectrogram.
- `output_audio.wav`: Edited audio file.
- `output_spec.png`: Output spectrogram.

#### Using Examples in Gradio:
The Gradio demo automatically loads examples from the `Examples` directory. Select an example to populate the interface with predefined parameters and files.

#### Example Directory Structure:
```
Examples/
└── Piano_and_Drums/
    └── Edit_Task/
        ├── hyper_params.json
        ├── input_audio.wav
        ├── input_spec.png
        ├── output_audio.wav
        ├── output_spec.png
```

#### Example `hyper_params.json`:
```json
{
    "source_prompt": "A music loop with piano and drums",
    "edit_mode": "replace",
    "add_src": "flute",
    "remove_src": "drums",
    "steps": 100,
    "cfg_scale_src": 3,
    "cfg_scale_trg": 12,
    "mixed_sampling": true,
    "close_start": 40,
    "far_start": 60,
    "mix_start": 30,
    "mix_end": 32,
    "randomize_seed": false
}
```

---

## Supported Edit Modes

- **Addition:** Add a semantic attribute to the audio (e.g., add "flute" to a piano loop).
- **Delete:** Remove a semantic attribute from the audio (e.g., remove "drums" from a music loop).
- **Replace:** Replace one attribute with another (e.g., replace "piano" with "violin").
- **Style Transfer:** Apply a specific style to the audio (e.g., "A Jazz style music loop").

---

## Supported Sources

- `accordion`
- `acoustic guitar`
- `bass guitar`
- `drums`
- `flute`
- `harmonica`
- `piano`
- `trumpet`
- `violin`
- `xylophone`

---

## Citation

TO BE ADDED

---

## License

TO BE ADDED

---

## References

This repository builds upon and integrates ideas and code from the following projects:

- [Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion](https://github.com/HilaManor/AudioEditingCode)
- [Audio Generation with AudioLDM](https://github.com/haoheliu/AudioLDM)
- [AudioSR: Versatile Audio Super-resolution at Scale](https://github.com/haoheliu/versatile_audio_super_resolution)

We thank the authors of these repositories for their contributions to the open-source community.