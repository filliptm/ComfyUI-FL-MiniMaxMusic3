# FL MiniMax Music 3

MiniMax Music 3 nodes for ComfyUI with verified model downloads, full audio VAE loading, MOSS-Music dataset preprocessing, and resumable LoRA training through a pinned SimpleTuner backend.

[![MiniMax Music 3](https://img.shields.io/badge/MiniMax-Music%203-blue?style=for-the-badge)](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
[![MOSS Music](https://img.shields.io/badge/MOSS-Music%208B-22c55e?style=for-the-badge)](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct)
[![Patreon](https://img.shields.io/badge/Patreon-Support%20Me-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/Machinedelusions)

## Features

- **Music 3 Loader** - Downloads, verifies, and loads the MODEL, CLIP, and decoder VAE with live per-file progress
- **Full Audio VAE** - Loads the SimpleTuner DAV encoder/decoder for waveform encoding and decoding
- **Intelligent Dataset Preprocessing** - Uses MOSS-Music to caption, transcribe, analyze, and segment local music libraries
- **Large Library Support** - Reads an absolute source directory at execution time without uploading or copying the source collection
- **LoRA Training** - Runs the pinned Music 3 SimpleTuner integration in an isolated environment
- **Live Training Dashboard** - On-node step, loss, learning-rate, progress, loss-chart, and validation-audio monitoring
- **Durable Runs** - Preserves logs, metrics, checkpoints, validation samples, interruption state, and resume metadata
- **ComfyUI LoRA Export** - Copies completed adapters into the standard ComfyUI LoRA folder

## Nodes

| Node | Description |
|------|-------------|
| **FL MiniMax Music 3 Loader** | Downloads and loads the Music 3 diffusion model, text encoder, and decoder VAE |
| **FL MiniMax Music 3 Audio VAE Loader** | Loads the full waveform encoder/decoder DAV |
| **FL MiniMax Music 3 Dataset Preprocessor** | Creates captioned, transcribed, segmented Music 3 datasets with MOSS-Music |
| **FL MiniMax Music 3 Dataset** | Validates audio, captions, lyrics, durations, and dataset identity |
| **FL MiniMax Music 3 Train Config** | Defines LoRA rank, precision, optimizer, steps, and checkpoint behavior |
| **FL MiniMax Music 3 Validation Config** | Defines deterministic validation caption, lyrics, duration, seed, CFG, and sampling settings |
| **FL MiniMax Music 3 Training Run** | Loads a persisted run for inspection or resume |
| **FL MiniMax Music 3 LoRA Trainer** | Trains, monitors, checkpoints, resumes, and exports a Music 3 LoRA |

## Installation

### ComfyUI Manager

After the pack is listed in the Comfy Registry, search for **FL MiniMax Music 3** and install it.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/filliptm/ComfyUI-FL-MiniMaxMusic3.git
cd ComfyUI-FL-MiniMaxMusic3
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Quick Start

### Music 3 Inference

1. Add **FL MiniMax Music 3 Loader**
2. Choose the model precision, text-encoder precision, CLIP device, and download policy
3. Connect its MODEL, CLIP, and VAE outputs to the standard MiniMax Music 3 workflow
4. Queue the workflow; missing artifacts download only when the node executes
5. Monitor verification, download, and loading progress directly on the node

### Dataset Preprocessing

1. Put `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, or `.aac` files in a local directory
2. Add **FL MiniMax Music 3 Dataset Preprocessor**
3. Paste the directory's absolute path into `source_folder`
4. Set `backend_policy` and `model_policy` to their install/download options for the first run
5. Queue the node and monitor MOSS installation, model download, analysis, and segmentation in its dashboard
6. Connect the resulting dataset directly to the training nodes or review the generated sidecars first

The source directory is scanned recursively at execution time. Files are read in place and are not uploaded through the browser.

### LoRA Training

1. Prepare a dataset manually or with **FL MiniMax Music 3 Dataset Preprocessor**
2. Connect **Dataset** -> **Train Config** -> **LoRA Trainer**
3. Optionally connect **Validation Config** for checkpoint audio samples
4. Set `backend_policy` to `install_pinned_if_missing` for the first run
5. Queue the workflow and monitor the on-node training dashboard
6. Use **FL MiniMax Music 3 Training Run** to resume an interrupted run from its latest full checkpoint

Completed adapters are saved to:

```text
ComfyUI/models/loras/MiniMaxMusic3/<output-name>/<output-name>.safetensors
```

Load them with ComfyUI's standard **Load LoRA (Model Only)** node between the Music 3 loader and sampler.

## Dataset Format

Trainer-ready datasets contain one audio file and two text sidecars per segment:

```text
dataset/
  track_001.wav
  track_001.txt
  track_001.lyrics
  track_001.music3.json
```

- `.txt` contains the musical caption
- `.lyrics` contains lyrics and structure tags such as `[verse]` and `[chorus]`
- `.music3.json` contains provenance, source hashes, analysis, and review state for generated segments

The preprocessor writes lossless PCM WAV segments so Windows training does not depend on TorchCodec audio decoding.

## Models and Backends

| Component | Source | Purpose |
|-----------|--------|---------|
| MiniMax Music 3 | `MiniMaxAI/MiniMax-Music3` | Diffusion model, text encoder, and decoder VAE |
| Music 3 Encoder | `SimpleTuner/MiniMax-Music-3-Encoder` | Full waveform encoder/decoder DAV |
| MOSS-Music 8B Instruct | `OpenMOSS-Team/MOSS-Music-8B-Instruct` | Audio captioning, transcription, structure analysis, and segmentation |
| SimpleTuner | Pinned commit in `training/backend_manifest.json` | Music 3 LoRA training |

Inference models are stored in the normal ComfyUI model directories. MOSS-Music is checksum-verified under:

```text
ComfyUI/models/moss_music/MOSS-Music-8B-Instruct/
```

Preprocessing and training dependencies are installed into separate managed environments under:

```text
ComfyUI/user/fl_minimax_music3/backends/
```

They are not installed into ComfyUI's Python environment. Model and backend downloads begin only after an explicit queued node execution or dashboard action.

## Runs and Recovery

Preprocessing and training runs are persisted under:

```text
ComfyUI/user/fl_minimax_music3/
```

Interrupting ComfyUI requests a graceful worker exit and preserves the latest usable checkpoint. Resume validation prevents accidental changes to dataset identity, LoRA shape, precision, or output name.

## Requirements

- A current ComfyUI build with MiniMax Music 3 support
- Python 3.10+ for the node pack; the pinned training backend currently requires Python 3.12 or 3.13
- FFmpeg and FFprobe available on `PATH`
- NVIDIA CUDA GPU with BF16 support for the initial training backend
- At least 20 GiB free for training-backend installation, plus model and checkpoint storage
- Approximately 16.9 GiB for the optional MOSS-Music model

The first training release targets one NVIDIA GPU. Multi-GPU, LyCORIS, full-rank training, AnyFlow, TwinFlow, CREPA, and LayerSync are not currently exposed.

## Example Workflow

Start with [`example_workflows/MiniMax Music 3 LoRA Training.json`](example_workflows/MiniMax%20Music%203%20LoRA%20Training.json). Its deliberately short training and validation settings are intended for installation checks; increase them for real training.

## Development Tests

From the ComfyUI directory:

```powershell
D:\ComfyUI\venv\Scripts\python.exe -m unittest discover -s custom_nodes\ComfyUI-FL-MiniMaxMusic3\tests -q
node --test custom_nodes\ComfyUI-FL-MiniMaxMusic3\tests\*.mjs
```

## License

[Apache-2.0](LICENSE). The optional training backend installs and invokes unmodified [SimpleTuner](https://github.com/bghira/SimpleTuner) as a separate AGPL-3.0-or-later program. MiniMax Music 3, the encoder, and MOSS-Music artifacts retain their respective model licenses. See [NOTICE](NOTICE) for details.
