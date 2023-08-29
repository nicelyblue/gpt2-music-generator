from mido import MidiFile, MidiTrack, Message
from typing import Tuple, Union, Dict, List
from itertools import count
import pickle
import os

class MidiTokenizer:
    def __init__(self, base_path: str, sequence_length: int):
        self.base_path = base_path
        self.sequence_length = sequence_length
        self.vocab: Dict[Tuple[int, int, int], int] = {}
        self.tokens: List[int] = []
        self.counter = count()
        
    def find_nearest_known_token(self, oov_token: int) -> int:
        return min(self.vocab.values(), key=lambda k: abs(k - oov_token))
    
    def get_token(self, note: int, velocity: int, duration: int) -> int:
        token_key = (note, velocity, duration)
        if token_key not in self.vocab:
            self.vocab[token_key] = next(self.counter)
        return self.vocab[token_key]

    def process_midi(self, midi_path: str):
        midi = MidiFile(midi_path)
        for track in midi.tracks:
            current_time = 0
            note_on_time: Dict[int, Tuple[int, int]] = {}
            for msg in track:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    note_on_time[msg.note] = (current_time, msg.velocity)
                elif (msg.type == "note_on" and msg.velocity == 0) and msg.note in note_on_time:
                    note, velocity = msg.note, note_on_time[msg.note][1]
                    duration = current_time - note_on_time[msg.note][0]
                    self.tokens.append(self.get_token(note, velocity, duration))
                    del note_on_time[msg.note]

    def process_midi_files(self):
        file_names = [f for f in os.listdir(self.base_path) if f.endswith(".mid")]
        for file_name in file_names:
            midi_path = os.path.join(self.base_path, file_name)
            try:
                self.process_midi(midi_path)
            except Exception as e:
                print(f"Failed to process {midi_path}: {e}")

        num_full_sequences = len(self.tokens) // self.sequence_length
        trimmed_length = num_full_sequences * self.sequence_length
        self.tokens = self.tokens[:trimmed_length]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)