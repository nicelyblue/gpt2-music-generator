import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm


# Parameters
note_vocab_size = 128
relative_onset_vocab_size = 100
duration_vocab_size = 100
vocab_size = note_vocab_size + relative_onset_vocab_size + duration_vocab_size
sequence_length = 300
embedding_size = 256
num_heads = 16
num_layers = 8
num_epochs = 100
batch_size = 64
lr = 0.001

folder_path = "MIDI dataset"
training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_names = [f for f in os.listdir(folder_path) if f.endswith(".mid")]
midi_data = []

for file_name in file_names:
    midi_path = os.path.join(folder_path, file_name)
    midi = MidiFile(midi_path)

    events = []
    for track in midi.tracks:
        current_time = 0
        note_on_time = {}
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_on_time[msg.note] = current_time
            elif (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)) and msg.note in note_on_time:
                note = msg.note
                relative_onset = min(current_time - note_on_time[msg.note], relative_onset_vocab_size - 1)
                duration = min(current_time, duration_vocab_size - 1)
                
                events.append(note)
                events.append(note_vocab_size + relative_onset)
                events.append(note_vocab_size + relative_onset_vocab_size + duration)
                
                del note_on_time[msg.note]

    midi_data.extend(events[:sequence_length * (len(events) // sequence_length)])


save_directory = "gpt2_midi_model"
if os.path.exists(save_directory):
    model = GPT2LMHeadModel.from_pretrained(save_directory).to(device)
    training = False

if training:
    midi_data = torch.tensor(midi_data).view(-1, sequence_length)
    dataset = TensorDataset(midi_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=sequence_length,
        n_embd=embedding_size,
        n_layer=num_layers,
        n_head=num_heads,
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        total_batches = len(data_loader)

        for batch in tqdm(data_loader):
            inputs = batch[0][:, :-1].to(device)
            targets = batch[0][:, 1:].to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / total_batches
        print(f"Training Loss: {avg_loss:.4f}")

    model.eval()
    model.save_pretrained(save_directory)

generated = [torch.randint(0, vocab_size, (1,)).item()]
temperature = 0.2

for i in range(sequence_length):
    inputs = torch.tensor(generated[-3:], dtype=torch.long).unsqueeze(0).to(device)
    outputs = model(inputs).logits
    probs = F.softmax(outputs / temperature, dim=-1)
    next_token = torch.multinomial(probs[-1], 1)[0].item()
    generated.append(next_token)

midi = MidiFile()
track = MidiTrack()
midi.tracks.append(track)

for i in range(0, len(generated)-2, 3):  
    note = generated[i]
    relative_onset = generated[i+1] - note_vocab_size
    duration = generated[i+2] - (note_vocab_size + relative_onset_vocab_size)

    note = max(0, min(127, note))
    
    track.append(Message("note_on", note=note, velocity=64, time=relative_onset))
    track.append(Message("note_off", note=note, velocity=64, time=duration))

midi.save("generated_midi.mid")
