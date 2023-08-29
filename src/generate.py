import os
import json
import torch
import argparse
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from mido import MidiFile, MidiTrack, Message
from midi_tokenizer import MidiTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Music Generation with Transformers')
    parser.add_argument('model', type=str, help='Path to the saved model')
    parser.add_argument('--save_directory', default='output', type=str, help='Directory to save the generated MIDI')
    parser.add_argument('--filename', default='generated', type=str, help='MIDI filename')
    args = parser.parse_args()
    return args

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return config_data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()

    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()

    tokenizer = MidiTokenizer.load(os.path.join(args.model, 'tokenizer.pkl'))

    config = load_model_config(os.path.join(args.model, 'config.json'))

    generated = [torch.randint(0, config['vocab_size'], (1,)).item()]
    temperature = 0.6

    for _ in range(config['n_positions']):
        inputs = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        outputs = model(inputs).logits
        probs = F.softmax(outputs / temperature, dim=-1)
        next_token = torch.multinomial(probs[-1], 1)[0].item()

        if next_token not in tokenizer.vocab.values():
            next_token = tokenizer.handle_oov_token(next_token)

        generated.append(next_token)

    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for token in generated:
        note, velocity, duration = [k for k, v in tokenizer.vocab.items() if v == token][0]
        track.append(Message("note_on", note=note, velocity=velocity, time=0))
        track.append(Message("note_on", note=note, velocity=0, time=duration))

    midi.save('generated_midi.mid')

if __name__ == '__main__':
    main()