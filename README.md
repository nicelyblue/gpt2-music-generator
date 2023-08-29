# gpt2-music-generator
Description

This project aims to generate music using Transformer models. The model is trained on a dataset of MIDI files and learns to produce musical sequences, which can be rendered into actual audio or MIDI files.
Table of Contents

    Requirements
    Quick Start
    Code Structure
    Contributing

Requirements

    Python 3.x
    PyTorch
    Transformers library
    tqdm
    mido
    FluidSynth (for synthesizing MIDI to WAV)

Run the following to install the required packages:

pip install -r requirements.txt

Quick Start

    Clone this repository:

    bash

git clone https://github.com/nicelyblue/gpt2-music-generator

Navigate to the directory:

bash

cd gpt2-music-generator

Install the required Python packages:

pip install -r requirements.txt

Train the model:

python train.py model.json

Optional arguments:

    --data: Path to MIDI dataset folder (default: 'dataset')
    --save_directory: Directory to save the model (default: 'model')
    --num_epochs: Number of epochs for training (default: 2000)
    --batch_size: Batch size (default: 16)
    --learn_rate: Initial learning rate (default: 0.01)

Generate music:

python generate.py model

Optional arguments:

    --save_directory: Directory to save the generated MIDI (default: 'output')
    --filename: MIDI filename (default: 'generated')

Convert MIDI to WAV:

css

python synth.py <input-midi-file>

Optional arguments:

    --wav_file: Path to the output WAV file

View MIDI file details:

css

    python view.py <path-to-midi-file>

Code Structure

    train.py: Training script
    generate.py: Music generation script
    synth.py: Script to convert MIDI files to WAV
    view.py: Script to view details of a MIDI file
    midi_tokenizer.py: Module for MIDI tokenization
    model.json: Model configuration file

Model Configuration

Example model.json:

json

{
    "sequence_length": 20,
    "embedding_size": 4,
    "num_heads": 1,
    "num_layers": 1,
    "resid_pdrop": 0.2,
    "embd_pdrop": 0.2
}

Contributing

Feel free to submit issues or pull requests to improve the project.  