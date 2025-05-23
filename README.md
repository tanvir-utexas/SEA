# Caption-Free Diffusion Audio Editing via Semantic Embedding Arithmetic


## Overview

This repository provides code and tools for Caption-Free Diffusion Audio Editing using Semantic Embedding Arithmetic. The approach enables flexible audio editing without the need for textual captions.

## Features

- **Gradio Demo:** Interactive web interface for real-time audio editing.
- **Inference Script:** Command-line tool for batch audio editing.

## Installation

```bash

pip install -r requirements.txt
```

## Usage

### Gradio Demo

Launch the Gradio demo to interactively edit audio files:

```bash
python gradio_demo.py
```

Open the provided URL in your browser to use the interface.

### Inference

Run inference on your audio files using the command line:

```bash
python inference.py --input path/to/input.wav --output path/to/output.wav --edit_type <edit_type> --source <source_type>
```

- `--input`: Path to the input audio file (e.g., `.wav`).
- `--output`: Path to save the edited audio file.
- `--edit_type`: Type of editing operation to apply.
- `--source`: The audio source to edit (e.g., vocals, drums, bass, etc.).

#### Supported Edit Types

- `add`: Add a semantic attribute to the selected source (e.g., add "reverb" to vocals).
- `remove`: Remove a semantic attribute from the source (e.g., remove "echo" from drums).
- `replace`: Replace one attribute with another (e.g., replace "soft" with "bright" on guitar).
- `swap`: Swap attributes between two sources (e.g., swap "dry" from vocals with "wet" from drums).

#### Supported Sources

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

#### Example Usage

```bash
python inference.py --input samples/song.wav --output results/edited_song.wav --edit_type add --source vocals
```

This command adds the specified semantic attribute to the vocals in `song.wav` and saves the result to `edited_song.wav`.

Refer to the documentation for a full list of supported attributes and advanced options.
