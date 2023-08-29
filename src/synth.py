import os
import argparse
from midi2audio import FluidSynth

def midi_to_wav(midi_file, wav_file):
    sf2_path = os.path.join('lib', 'GeneralUser GS v1.471.sf2')
    
    fs = FluidSynth(sf2_path)
    fs.midi_to_audio(midi_file, wav_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MIDI to WAV.')
    parser.add_argument('midi_file', help='Path to the input MIDI file')
    parser.add_argument('--wav_file', help='Path to the output WAV file', default=None)

    args = parser.parse_args()

    midi_file = args.midi_file
    wav_file = args.wav_file if args.wav_file else os.path.splitext(midi_file)[0] + '.wav'

    midi_to_wav(midi_file, wav_file)