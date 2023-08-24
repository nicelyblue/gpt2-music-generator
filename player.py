import pygame
import time

def play_midi(file_path):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

if __name__ == '__main__':
    midi_file = 'generated_midi.mid'
    play_midi(midi_file)