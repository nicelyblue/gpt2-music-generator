import pygame
import time
from midi2audio import FluidSynth

def play_midi(file_path):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

def midi_to_wav(midi_file, wav_file):
    sf2_path = r'GeneralUser GS v1.471.sf2'
    
    fs = FluidSynth(sf2_path)
    fs.midi_to_audio(midi_file, wav_file)

if __name__ == '__main__':
    midi_file = 'generated_midi_2.mid'
    wav_file = 'generated_music_2.wav'

    midi_to_wav(midi_file, wav_file)

    # play_midi(midi_file)
