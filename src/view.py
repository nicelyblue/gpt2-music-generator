import mido
from mido import MidiFile
import argparse

parser = argparse.ArgumentParser(description='View MIDI file.')
parser.add_argument('path', help='Path to the MIDI file')

args = parser.parse_args()

mid = MidiFile(args.path)

for i, track in enumerate(mid.tracks):
    print(f"Track {i}: {track.name}")
    for msg in track:
        print(msg)