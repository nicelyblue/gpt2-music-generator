from midi2audio import FluidSynth

def midi_to_wav(midi_file, wav_file):
    sf2_path = r'GeneralUser GS v1.471.sf2'
    
    fs = FluidSynth(sf2_path)
    fs.midi_to_audio(midi_file, wav_file)

if __name__ == '__main__':
    midi_file = 'generated_midi.mid'
    wav_file = 'generated_music_4.wav'

    midi_to_wav(midi_file, wav_file)
